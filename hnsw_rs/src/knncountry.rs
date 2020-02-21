use std::collections::HashMap;
use crate::knnservice::KnnService;
use std::path::{Path, PathBuf};
use failure::Error;
use std::fs;
use crate::{Distance, IndexConfig};

#[derive(Default)]
pub struct KnnByCountry {
    countries: HashMap<String, KnnService>
}

const DIMENSION_FILENAME: &str = "_dimension";
const METRIC_FILENAME: &str = "_metrics";
const DEFAULT_EF_SEARCH: usize = 50;

impl KnnByCountry {
    pub fn load<P: AsRef<Path>>(&mut self, country: &str, indices_path: P, extra_item_path: P) -> Result<(), Error> {
        let indices_path = indices_path.as_ref();
        let extra_item_path = extra_item_path.as_ref();
        let index_country_root: PathBuf = PathBuf::from(indices_path).join(format!("country={}", country));
        let extra_country_root: PathBuf = PathBuf::from(extra_item_path).join(format!("country={}/non-recommendable", country));

        let dimension_path = index_country_root.clone().join(DIMENSION_FILENAME);
        let metric_path = index_country_root.clone().join(METRIC_FILENAME);

        let dimension_value: usize = fs::read_to_string(dimension_path)?.parse()?;
        let metric_value:Distance = fs::read_to_string(metric_path)?.parse()?;

        let mut knn_service = KnnService::new(IndexConfig::new(metric_value, dimension_value, DEFAULT_EF_SEARCH));
        knn_service.load(index_country_root, extra_country_root)?;
        self.countries.insert(country.to_string(), knn_service);
        Ok(())
    }

    pub fn get_service(&self, country: &str) -> Option<&KnnService> {
        self.countries.get(country)
    }

    pub fn get_countries(&self) -> Vec<String> {
        self.countries.keys().cloned().collect()
    }
}