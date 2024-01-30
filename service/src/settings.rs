use config::{Config, Environment, File};
use serde::Deserialize;
use std::{env, path::PathBuf};

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerConfig {
    pub worker_thread: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexConfig {
    pub embedding_version: String,
    pub indices_root: PathBuf,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub name: String,
    pub path: Option<PathBuf>,
    pub model_type: String,
    pub is_default: bool,
    pub version: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelConfig {
    pub models: Vec<Model>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnnConfig {
    pub server: ServerConfig,
    pub index_config: IndexConfig,
    pub model_config: ModelConfig,
    pub platform: String,
    pub countries: Vec<String>,
}

impl KnnConfig {
    pub fn new() -> anyhow::Result<Self> {
        let run_mode = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());
        let s = Config::builder()
            .add_source(File::with_name("config/default.toml"))
            .add_source(File::with_name(&format!("config/{}", run_mode)).required(false))
            .add_source(Environment::with_prefix("KNN"))
            .build()?;

        let res = s.try_deserialize()?;
        Ok(res)
    }
}
