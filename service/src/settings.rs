use std::env;
use config::{ConfigError, Config, File};

#[derive(Debug, Deserialize)]
pub struct Graphite {
    pub endpoint: String,
    pub prefix: String,
}

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub country: Vec<String>,
    pub graphite: Option<Graphite>
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let mut s = Config::new();
        s.merge(File::with_name("config/default"))?;
        let env = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());
        s.merge(File::with_name(&format!("config/{}", env)).required(false))?;
        s.try_into()
    }
}