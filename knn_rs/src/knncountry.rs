use crate::knnservice::{KnnService, Model};
use crate::{Distance, IndexConfig, KnnError};
use log::{error, info, warn};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone)]
struct KnnCountryConfig {
    indices_root_path: PathBuf,
    models: Vec<Model>,
    platform: String,
    version: String,
}

impl KnnCountryConfig {
    fn indice_path(&self, country: &str) -> PathBuf {
        self.indices_root_path
            .join(self.platform.clone())
            .join(self.version.clone())
            .join(format!("country={}", country))
    }
}

#[derive(Default)]
pub struct KnnByCountry {
    config: KnnCountryConfig,
    countries: HashMap<String, KnnService>,
}

const DIMENSION_FILENAME: &str = "_dimension";
const METRIC_FILENAME: &str = "_metrics";
const DEFAULT_EF_SEARCH: usize = 50;

impl KnnByCountry {
    pub fn new(config: KnnCountryConfig) -> KnnByCountry {
        KnnByCountry {
            config,
            countries: HashMap::new(),
        }
    }

    pub fn load(&mut self, country: &str) -> Result<(), KnnError> {
        let mut knn_service = KnnService::new();
        knn_service.load_index(self.config.indice_path(country))?;

        for m in self.config.models.iter() {
            let mpath = m
                .model_path
                .join(self.config.platform.clone())
                .join(m.version.clone())
                .join(format!("country={}", country));
            knn_service.load_model(m.clone(), mpath)?;
        }
        self.countries.insert(country.to_string(), knn_service);
        Ok(())
    }

    fn load_countries(&mut self, countries: &[String]) -> Result<(), KnnError> {
        for c in countries {
            self.load(c)?
        }
        Ok(())
    }

    pub fn get_service(&self, country: &str) -> Option<&KnnService> {
        self.countries
            .get(country)
            .or_else(|| self.countries.get("XX"))
    }

    pub fn get_countries(&self) -> Vec<String> {
        self.countries.keys().cloned().collect()
    }
}
