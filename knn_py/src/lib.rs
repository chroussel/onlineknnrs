extern crate hnsw_rs;
extern crate pyo3;

use pyo3::prelude::*;
use pyo3::exceptions::TypeError;
use hnsw_rs::*;
use hnsw_rs::knnservice::Model;
use env_logger::Env;

#[pyclass(module = "knn_service")]
struct KnnService {
    knn_country: knncountry::KnnByCountry
}

#[pymethods]
impl KnnService {
    #[new]
    fn new() -> Self {
        let knn_country = knncountry::KnnByCountry::default();
        KnnService {
            knn_country
        }
    }

    fn load_country(&mut self, country: String, index_path: String, embedding_path: String) -> PyResult<()> {
        self.knn_country.load(&country, index_path, embedding_path)
            .map_err(|e| PyErr::new::<TypeError, _>(e.to_string()))
    }

    fn query(&self, country: &str, index: i32, result_count: usize, timeline: Vec<(i32, i64)>) -> PyResult<Vec<(i64, f32)>> {
        if let Some(service) = self.knn_country.get_service(country) {
           service.get_closest_items(&timeline, index, result_count, Model::Average)
               .map_err(|e| PyErr::new::<TypeError, _>(e.to_string()))
        } else {
            Ok(vec![])
        }
    }
}


#[pymodule]
fn knn_py(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::from_env(Env::default().default_filter_or("info")).init();
    m.add_class::<KnnService>()?;
    Ok(())
}