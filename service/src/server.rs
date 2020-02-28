#[macro_use] extern crate log;
extern crate env_logger;
extern crate hnsw_rs;
extern crate dipstick;
extern crate hdrhistogram;
extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate metrics_runtime;
extern crate metrics_core;
extern crate clokwerk;

mod settings;
mod knn;
mod knn_controller;
mod metric_observer;

use std::path::PathBuf;
use failure::Error;
use clap::{App, Arg};
use dipstick::{Graphite, Prefixed, ScheduleFlush, AtomicBucket, Stream};
use env_logger::Env;
use crate::knn_controller::KnnController;
use tonic::transport::Server;
use crate::knn::knn_server::*;
use std::time::Duration;
use settings::Settings;
use crate::metric_observer::GraphiteObserver;
use metrics_runtime::Receiver;
use clokwerk::{Scheduler, Interval};
use metrics_core::Observe;


struct ResultArgs {
    port: u16,
    index_path: PathBuf,
    extra_item_path: PathBuf
}

fn parse_args() -> Result<ResultArgs, Error> {
    let matches = App::new("Knn Serving")
        .arg(Arg::with_name("port").short("p").long("port").value_name("PORT").takes_value(true).required(true))
        .arg(Arg::with_name("index_path").short("i").long("index_path").value_name("PATH").takes_value(true).required(true))
        .arg(Arg::with_name("extra_path").short("e").long("extra_path").value_name("PATH").takes_value(true).required(true))
        .get_matches();

    let port: u16 = matches.value_of("port").unwrap().parse()?;
    let index_path: PathBuf = PathBuf::from(matches.value_of("index_path").unwrap());
    let extra_item_path: PathBuf = PathBuf::from(matches.value_of("extra_path").unwrap());

    Ok(ResultArgs {
        port,
        index_path,
        extra_item_path
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>>{
    env_logger::from_env(Env::default().default_filter_or("info")).init();
    let settings = Settings::new()?;
    let app_metrics = AtomicBucket::new();

    if let Some(graphite_settings) = settings.graphite {
        info!("Using graphite with endpoint {} and prefix {}", graphite_settings.endpoint, graphite_settings.prefix);
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
    scheduler.every(Interval::Seconds(10)).run(move || controller.observe(&mut observer));

    let handler = scheduler.watch_thread(Duration::from_millis(500));
    let result_args = parse_args()?;
    let addr = format!("0.0.0.0:{}", result_args.port).parse().unwrap();
    info!("Initializing server");
    let mut controller = KnnController::new(settings.country, app_metrics, &receiver);
    controller.load(&result_args.index_path, &result_args.extra_item_path)?;

    info!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(controller))
        .serve(addr)
        .await?;

    handler.stop();
    info!("Stopping the server");
    Ok(())
}
