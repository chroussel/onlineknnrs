use std::collections::HashMap;
use crate::knnservice::KnnService;
use std::path::{Path, PathBuf};
use failure::Error;
use std::fs::File;
use std::io::Read;
use std::fs;
use crate::{Distance, IndexConfig};

#[derive(Default)]
pub struct KnnByCountry {
    countries: HashMap<String, KnnService>
}

const dimensionFilename: &str = "_dimension";
const metricFilename: &str = "_metrics";
const defaultEfSearch: usize = 50;

impl KnnByCountry {
    pub fn load<P: AsRef<Path>>(&mut self, country: &str, indices_path: P, extra_item_path: P) -> Result<(), Error> {
        let indices_path = indices_path.as_ref();
        let extra_item_path = extra_item_path.as_ref();
        let index_country_root: PathBuf = PathBuf::from(indices_path).join(format!("country={}", country));
        let extra_country_root: PathBuf = PathBuf::from(extra_item_path).join(format!("country={}/non-recommendable", country));

        let dimensionPath = index_country_root.clone().join(dimensionFilename);
        let metricPath = index_country_root.clone().join(metricFilename);

        let dimensionValue: usize = fs::read_to_string(dimensionPath)?.parse()?;
        let metricValue:Distance = fs::read_to_string(metricPath)?.parse()?;

        let mut knn_service = KnnService::new(IndexConfig::new(metricValue, dimensionValue, defaultEfSearch));
        knn_service.load(index_country_root, extra_country_root);
        self.countries.insert(country.to_string(), knn_service);
        Ok(())
    }

    pub fn get_service(&self, country: &str) -> Option<&KnnService> {
        self.countries.get(country)
    }

    pub fn get_countries(&self) -> Vec<String> {
        self.countries.keys().map(|s| s.clone()).collect()
    }
}