#[macro_use]
extern crate tracing;

mod knn;
mod knn_controller;
mod settings;

use crate::knn::knn_server::*;
use crate::knn_controller::KnnController;
use anyhow::anyhow;
use clap::Parser;
use knn_rs::knncountry::KnnConfig;
use metrics_exporter_prometheus::PrometheusBuilder;
use settings::Settings;
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};
use tonic::transport::Server;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(clap::Parser)]
#[command()]
struct KnnServiceArgs {
    #[arg(long = "http_port", value_name = "PORT")]
    http_port: u16,
    #[arg(long = "grpc_port", value_name = "PORT")]
    grpc_port: u16,
    #[arg(long = "index_root_path", value_name = "PATH")]
    index_root_path: PathBuf,
    #[arg(long = "model_root_path", value_name = "PATH")]
    model_root_path: Option<PathBuf>,
    #[arg(short = 'c', long = "core_count", value_name = "CORE_COUNT")]
    core_count: Option<usize>,
    #[arg(long = "platform", value_name = "PLATFORM")]
    platform: String,
    #[arg(long = "version", value_name = "VERSION")]
    version: String,
}

fn setup(http_port: u16) -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::ERROR.into())
                .from_env_lossy(),
        )
        .init();
    let addr: SocketAddr = format!("0.0.0.0:{}", http_port).parse().unwrap();
    let builder = PrometheusBuilder::new();
    let handle = builder.with_http_listener(addr).install()?;

    Ok(())
}

async fn run(settings: Settings, args: KnnServiceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{}", args.grpc_port).parse().unwrap();
    info!("Initializing server");

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<KnnServer<KnnController>>()
        .await;

    let indices_root_path = expand(&args.index_root_path)?;
    let mut model_path = None;
    if let Some(mp) = args.model_root_path {
        let res = expand(mp)?;
        model_path = Some(res);
    }
    let config = KnnConfig {
        indices_root_path: indices_root_path.clone(),
        models: vec![],
        platform: args.platform,
        version: args.version,
    };

    let mut controller = KnnController::new(settings.country, config);
    controller.load()?;

    info!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(controller))
        .serve(addr)
        .await?;

    info!("Stopping the server");
    Ok(())
}

fn expand<P: AsRef<Path>>(path: P) -> anyhow::Result<PathBuf> {
    let path = path.as_ref();
    let path_str = path.as_os_str().to_str().ok_or(anyhow!("Invalid path"))?;
    let new_path = PathBuf::from(shellexpand::tilde(path_str).to_string());
    Ok(new_path)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = KnnServiceArgs::parse();
    setup(args.http_port)?;
    let settings = Settings::new().expect("Unable to parse settings");

    let mut rt_builder = tokio::runtime::Builder::new_multi_thread();
    if let Some(core_count) = args.core_count {
        rt_builder.worker_threads(core_count);
    }
    let mut rt = rt_builder.enable_all().build().unwrap();
    rt.block_on(run(settings, args)).expect("Working");
    Ok(())
}
