extern crate hnsw_rs;
extern crate pyo3;
extern crate shellexpand;

use pyo3::prelude::*;
use pyo3::exceptions::TypeError;
use hnsw_rs::*;
use hnsw_rs::knnservice::Model;
use env_logger::Env;
use hnsw_rs::embedding_computer::UserEvent;
use pyo3::types::PyAny;

struct PyUserEvent(UserEvent);

impl From<(i32, i64, u64, i32)> for PyUserEvent {
    fn from((index, label, timestamp, event_type): (i32, i64, u64, i32)) -> Self {
        PyUserEvent(UserEvent {
            index, label, timestamp, event_type
        })
    }
}

#[pyclass]
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

    fn load_model(&mut self, country: String, model_name: String, model_path: String) -> PyResult<()> {
        let model_path = shellexpand::tilde(&model_path).to_string();
        self.knn_country.load_model(&country, &model_name, model_path)
            .map_err(|e| PyErr::new::<TypeError, _>(e.to_string()))
    }

    fn query(&self, country: &str, index: i32, result_count: usize, timeline: Vec<(i32, i64, u64, i32)>) -> PyResult<Vec<(i64, f32)>> {
        if let Some(service) = self.knn_country.get_service(country) {
            let user_events: Vec<UserEvent> = timeline.into_iter().map(PyUserEvent::from).map(|pu| pu.0).collect();
            service.get_closest_items(&user_events, index, result_count, Model::Average)
               .map_err(|e| PyErr::new::<TypeError, _>(e.to_string()))
        } else {
            Ok(vec![])
        }
    }

    fn tf_query(&self, country: &str, index: i32, result_count: usize, timeline: Vec<(i32, i64, u64, i32)>, model_name: String) -> PyResult<Vec<(i64, f32)>> {
        if let Some(service) = self.knn_country.get_service(country) {
            let user_events: Vec<UserEvent> = timeline.into_iter().map(PyUserEvent::from).map(|pu| pu.0).collect();
            service.get_closest_items(&user_events, index, result_count, Model::Tensorflow(model_name))
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