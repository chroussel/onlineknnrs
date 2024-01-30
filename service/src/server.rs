#[macro_use]
extern crate tracing;

mod knn;
mod knn_controller;
mod settings;

use crate::knn::knn_server::*;
use crate::knn_controller::KnnController;
use anyhow::anyhow;
use clap::Parser;
use knn_rs::knnservice::{Model, ModelType};
use metrics_exporter_prometheus::PrometheusBuilder;
use settings::KnnConfig;
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
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
}

fn setup_logging() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();

    Ok(())
}

fn setup_metrics(http_port: u16) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("0.0.0.0:{}", http_port).parse().unwrap();
    let builder = PrometheusBuilder::new();
    builder.with_http_listener(addr).install()?;
    Ok(())
}

async fn run(config: KnnConfig, args: KnnServiceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{}", args.grpc_port).parse().unwrap();
    info!("Initializing server");

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<KnnServer<KnnController>>()
        .await;

    let indices_root_path = expand(&config.index_config.indices_root)?;
    let models = config
        .model_config
        .models
        .iter()
        .map(|m| {
            let mut model_path = None;
            if let Some(mp) = &m.path {
                model_path = Some(expand(mp)?);
            }
            Ok(Model {
                name: m.name.clone(),
                model_path,
                model_type: ModelType::from_str(&m.model_type)?,
                is_default: m.is_default,
                version: m.version.clone(),
            })
        })
        .collect::<anyhow::Result<Vec<Model>>>()?;
    let config = knn_rs::knncountry::Config {
        models,
        indices_root_path: indices_root_path.clone(),
        platform: config.platform,
        version: config.index_config.embedding_version,
        countries: config.countries,
    };

    let mut controller = KnnController::new(config);
    controller.load()?;

    info!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(controller))
        .add_service(health_service)
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
    setup_logging()?;
    info!("Setup server");
    let config = KnnConfig::new()?;

    let mut rt_builder = tokio::runtime::Builder::new_multi_thread();
    if let Some(core_count) = config.server.worker_thread {
        rt_builder.worker_threads(core_count as usize);
    }
    let rt = rt_builder.enable_all().build().unwrap();
    setup_metrics(args.http_port)?;
    rt.block_on(run(config, args))?;
    Ok(())
}
