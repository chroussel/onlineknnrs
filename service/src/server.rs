#[macro_use] extern crate log;
extern crate env_logger;
extern crate hnsw_rs;
#[macro_use] extern crate dipstick;
extern crate hdrhistogram;
extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate metrics_runtime;
extern crate metrics_core;

mod settings;
mod knn;
mod knn_controller;
mod metric_observer;

use std::path::PathBuf;
use failure::Error;
use clap::{App, Arg};
use dipstick::{Graphite, Prefixed, Input, ScheduleFlush, Proxy, AtomicBucket, Stream};
use env_logger::Env;
use crate::knn_controller::KnnController;
use tonic::transport::Server;
use crate::knn::{*, knn_server::*};
use std::time::Duration;
use settings::Settings;


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
        app_metrics.drain(graphite)
    } else {
        info!("Graphite is disabled");
    }
    app_metrics.flush_every(Duration::new(60, 0));

    let result_args = parse_args()?;
    let addr = format!("0.0.0.0:{}", result_args.port).parse().unwrap();
    info!("Initializing server");
    let mut controller = KnnController::new(settings.country, app_metrics);
    controller.load(&result_args.index_path, &result_args.extra_item_path)?;

    info!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(controller))
        .serve(addr)
        .await?;
    Ok(())
}
