use std::collections::HashMap;
use crate::knnservice::{KnnService, Model};
use std::path::{Path, PathBuf};
use failure::{Error, ResultExt};
use std::fs;
use crate::{Distance, IndexConfig, KnnError};

#[derive(Default)]
pub struct KnnByCountry {
    countries: HashMap<String, KnnService>
}

const DIMENSION_FILENAME: &str = "_dimension";
const METRIC_FILENAME: &str = "_metrics";
const DEFAULT_EF_SEARCH: usize = 50;

impl KnnByCountry {
    pub fn load<P: AsRef<Path>>(&mut self, country: &str, indices_path: P, extra_item_path: P, models_path: Option<P>) -> Result<(), Error> {
        let indices_path = indices_path.as_ref();
        let extra_item_path = extra_item_path.as_ref();
        let models_path = models_path.map(|p| PathBuf::from(p.as_ref()).join(format!("country={}/_model.pb", country)));
        let index_country_root: PathBuf = PathBuf::from(indices_path).join(format!("country={}", country));
        let extra_country_root: PathBuf = PathBuf::from(extra_item_path).join(format!("country={}/non-recommendable", country));

        let dimension_path = index_country_root.clone().join(DIMENSION_FILENAME);
        let metric_path = index_country_root.clone().join(METRIC_FILENAME);

        let dimension_value: usize = fs::read_to_string(&dimension_path).context(format!("Error while reading {}", dimension_path.display()))?.parse()?;
        let metric_value: Distance = fs::read_to_string(&metric_path).context(format!("Error while reading {}", metric_path.display()))?.parse()?;

        let mut knn_service = KnnService::new(IndexConfig::new(metric_value, dimension_value, DEFAULT_EF_SEARCH));
        knn_service.load(index_country_root, extra_country_root)?;
        if let Some(models_root) = models_path {
            if models_root.exists() {
                knn_service.load_model(Model::Tensorflow("default".into()), models_root)?;
            } else {
                warn!("No model could be found in {}. Skipping", models_root.display());
            }
        }
        self.countries.insert(country.to_string(), knn_service);
        Ok(())
    }

    pub fn load_model<P: AsRef<Path>>(&mut self, country: &str, model_name: &str, model_path: P) -> Result<(), Error> {
        let service = self.countries.get_mut(country).ok_or_else(|| KnnError::CountryNotFoundWhileLoadingModel(country.to_string()))?;
        service.load_model(Model::Tensorflow(model_name.to_string()), model_path)
    }

    pub fn get_service(&self, country: &str) -> Option<&KnnService> {
        self.countries.get(country).or_else(|| self.countries.get("XX"))
    }

    pub fn get_countries(&self) -> Vec<String> {
        self.countries.keys().cloned().collect()
    }
}