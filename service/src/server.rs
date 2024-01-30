#[macro_use]
extern crate log;
extern crate dipstick;
extern crate env_logger;
extern crate hdrhistogram;
extern crate hnsw_rs;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate clokwerk;
extern crate metrics_core;
extern crate metrics_runtime;

mod knn;
mod knn_controller;
mod metric_observer;
mod settings;
//mod health_check;

use crate::knn::knn_server::*;
use crate::knn_controller::KnnController;
use crate::metric_observer::GraphiteObserver;
use clap::{App, Arg};
use clokwerk::{Interval, Scheduler};
use dipstick::{AtomicBucket, Graphite, Prefixed, ScheduleFlush, Stream};
use env_logger::Env;
use failure::Error;
use hnsw_rs::knnservice::Model;
use metrics_core::Observe;
use metrics_runtime::Receiver;
use settings::Settings;
use std::path::PathBuf;
use std::time::Duration;
use tokio::runtime;
use tonic::transport::Server;

struct ResultArgs {
    port: u16,
    index_path: PathBuf,
    extra_item_path: PathBuf,
    models_path: Option<PathBuf>,
    model: Model,
    core_count: Option<usize>,
}

fn parse_args() -> Result<ResultArgs, Error> {
    let matches = App::new("Knn Serving")
        .arg(
            Arg::with_name("port")
                .short("p")
                .long("port")
                .value_name("PORT")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("index_path")
                .short("ip")
                .long("index_path")
                .value_name("PATH")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("extra_path")
                .short("ep")
                .long("extra_path")
                .value_name("PATH")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("models_path")
                .short("mp")
                .long("models_path")
                .value_name("PATH")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("model")
                .long("model")
                .value_name("MODEL_NAME")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("core")
                .short("c")
                .long("core")
                .value_name("CORE_COUNT")
                .takes_value(true),
        )
        .get_matches();

    let port: u16 = matches.value_of("port").unwrap().parse()?;
    let core_count: Option<usize> = matches.value_of("core").and_then(|v| v.parse().ok());
    let index_path: PathBuf =
        PathBuf::from(shellexpand::tilde(matches.value_of("index_path").unwrap()).to_string());
    let extra_item_path: PathBuf =
        PathBuf::from(shellexpand::tilde(matches.value_of("extra_path").unwrap()).to_string());
    let models_path = matches
        .value_of("models_path")
        .map(|m| PathBuf::from(shellexpand::tilde(m).to_string()));
    let model = Model::from(matches.value_of("model").unwrap());
    Ok(ResultArgs {
        port,
        index_path,
        extra_item_path,
        models_path,
        model,
        core_count,
    })
}

async fn run(
    settings: Settings,
    result_args: ResultArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let app_metrics = AtomicBucket::new();

    if let Some(graphite_settings) = settings.graphite {
        info!(
            "Using graphite with endpoint {} and prefix {}",
            graphite_settings.endpoint, graphite_settings.prefix
        );
        let graphite = Graphite::send_to(graphite_settings.endpoint)
            .expect("Connected to graphite")
            .named(graphite_settings.prefix);
        app_metrics.drain(graphite);
        app_metrics.flush_every(Duration::new(60, 0));
    } else {
        info!("Graphite is disabled");
        app_metrics.drain(Stream::to_stdout());
        app_metrics.flush_every(Duration::new(5, 0));
    }

    let receiver = Receiver::builder()
        .histogram(Duration::from_secs(60), Duration::from_secs(1))
        .build()
        .expect("Working receiver");
    let mut observer = GraphiteObserver::new(app_metrics.clone());
    let controller = receiver.controller();
    controller.observe(&mut observer);
    let mut scheduler = Scheduler::new();
    scheduler
        .every(Interval::Seconds(10))
        .run(move || controller.observe(&mut observer));

    let handler = scheduler.watch_thread(Duration::from_millis(500));

    let addr = format!("0.0.0.0:{}", result_args.port).parse().unwrap();
    info!("Initializing server");

    let mut controller =
        KnnController::new(settings.country, app_metrics, &receiver, result_args.model);
    controller.load(
        &result_args.index_path,
        &result_args.extra_item_path,
        result_args.models_path.as_ref(),
    )?;

    info!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(controller))
        .serve(addr)
        .await?;

    handler.stop();
    info!("Stopping the server");
    Ok(())
}

fn main() {
    env_logger::from_env(Env::default().default_filter_or("info")).init();
    let settings = Settings::new().expect("Unable to parse settings");
    let result_args = parse_args().expect("Unable to parse cli arguments");

    let mut rt_builder = runtime::Builder::new();
    rt_builder.threaded_scheduler();
    if let Some(core_count) = &result_args.core_count {
        rt_builder.core_threads(*core_count);
    }
    let mut rt = rt_builder.enable_all().build().unwrap();
    rt.block_on(run(settings, result_args)).expect("Working");
}
