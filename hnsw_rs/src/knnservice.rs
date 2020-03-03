use std::path::{Path, PathBuf};
use ndarray::*;
use crate::*;
use crate::knnindex::{EmbeddingRegistry, KnnIndex};
use crate::loader::Loader;
use failure::Error;
use std::collections::HashMap;
use crate::embedding_computer::{UserEmbeddingComputer, AverageComputer, EmbeddingResult, UserEvent};
use std::fmt::Display;
use failure::_core::fmt::Formatter;
use crate::knn_tf::KnnTf;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum Model {
    Average,
    Tensorflow(String)
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Average => {
                f.write_str("Average")?;
            },
            Model::Tensorflow(model_name) => {
                f.write_str(model_name)?;
            },
        }
        Ok(())
    }
}

pub struct KnnService {
    config: IndexConfig,
    embedding_registry: EmbeddingRegistry,
    models: HashMap<Model, Box<dyn UserEmbeddingComputer>>
}

impl KnnService {
    pub fn new(config: IndexConfig) -> Self {
        let mut average: HashMap<Model, Box<dyn UserEmbeddingComputer>> = HashMap::new();
        average.insert(Model::Average, Box::new(AverageComputer::default()));
        KnnService {
            config,
            embedding_registry: EmbeddingRegistry::new(config.dim),
            models: average
        }
    }

    pub fn load<P: AsRef<Path>>(&mut self, indices_path: P, extra_item_path: P) -> Result<(), Error> {
        info!("KnnService: Starting load from {} and {}", indices_path.as_ref().display(), extra_item_path.as_ref().display());
        Loader::load_index_folder(indices_path, |index, path| self.add_index(index, path))?;
        Loader::load_extra_item_folder(extra_item_path, |index_id, label, embedding| self.add_extra_item(index_id, label, embedding))?;
        info!("KnnService: Load done");
        Ok(())
    }

    pub fn load_model<P: AsRef<Path>>(&mut self, model: Model, tf_model: P) -> Result<(), Error > {
        info!("KnnService: Starting model load from {}", tf_model.as_ref().display());
        let tf_computer = KnnTf::load_model(tf_model)?;
        self.models.insert(model, Box::new(tf_computer));
        info!("KnnService: Model load done");
        Ok(())
    }

    pub fn add_index(&mut self, index_id: i32, path: PathBuf) -> Result<(), Error> {
        let config = self.config;
        let index = self.embedding_registry.embeddings.entry(index_id)
            .or_insert_with(move || KnnIndex::new(config));
        index.add_index_from_path(path)
    }

    pub fn add_extra_item(&mut self, index_id: i32, label: i64, embedding: Array1<f32> ) {
        let config = self.config;
        let index = self.embedding_registry.embeddings.entry(index_id)
            .or_insert_with(move || KnnIndex::new(config));
        index.add_extra_item(label, embedding);
    }

    fn compute_user_vector(&self, model: Model, user_events: &[UserEvent]) -> Result<EmbeddingResult, Error> {
        self.models.get(&model)
            .ok_or(KnnError::ModelNotFound(model))
            .and_then(|m|
                m.compute_user_vector(&self.embedding_registry, user_events)
            )
            .map_err(From::from)
    }

    pub fn get_closest_items(&self, user_events: &[UserEvent], query_index: i32, k: usize, model: Model) -> Result<Vec<(i64, f32)>, Error> {
        let mut user_vector = self.compute_user_vector(model, user_events)?;

        if user_vector.user_event_used_count == 0 {
            return Ok(vec!())
        }
        Ok(self.embedding_registry.embeddings
            .get(&query_index)
            .map(|index| index.search(user_vector.user_embedding.view_mut(), k))
            .unwrap_or_else(|| vec!()))
    }
}

#[cfg(test)]
mod tests {
    use crate::knnservice::*;
    use crate::{Distance, IndexConfig};

    #[test]
    fn simple_test() {
        let config = IndexConfig::new(Distance::Euclidean, 100, 10);
        let a = KnnService::new(config);

    }
}
