use std::path::{Path, PathBuf};
use ndarray::*;
use crate::*;
use crate::knnindex::{EmbeddingRegistry, KnnIndex};
use crate::loader::Loader;
use failure::Error;

pub enum Model {
    Average
}

pub struct EmbeddingResult {
    user_embedding: Array1<f32>,
    user_event_used_count: usize
}

pub struct KnnService {
    config: IndexConfig,
    embedding_registry: EmbeddingRegistry
}

impl KnnService {
    pub fn new(config: IndexConfig) -> Self {
        KnnService {
            config,
            embedding_registry: EmbeddingRegistry::default()
        }
    }

    pub fn load<P: AsRef<Path>>(&mut self, indices_path: P, extra_item_path: P) -> Result<(), Error> {
        info!("KnnService: Starting load from {} and {}", indices_path.as_ref().display(), extra_item_path.as_ref().display());
        Loader::load_index_folder(indices_path, |index, path| self.add_index(index, path))?;
        Loader::load_extra_item_folder(extra_item_path, |index_id, label, embedding| self.add_extra_item(index_id, label, embedding))?;
        info!("KnnService: Load done");
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

    pub  fn compute_user_vector(&self, labels: &[(i32, i64)]) -> Result<EmbeddingResult, Error> {
        let mut count = 0;
        let mut user_vector = Array1::<f32>::zeros(self.config.dim);

        for &(index_id, label) in labels {
            if let Some(data_vector) = self.embedding_registry.fetch_item(index_id, label) {
                count += 1;
                user_vector += &data_vector
            }
        }
        if count != 0 {
            user_vector /= count as f32;
        }

        Ok(EmbeddingResult {
            user_embedding: user_vector,
            user_event_used_count: count
        })
    }

    pub fn get_closest_items(&self, labels: &[(i32, i64)], query_index: i32, k: usize, model: Model) -> Result<Vec<(i64, f32)>, Error> {
        let mut user_vector = match model {
            Model::Average => self.compute_user_vector(labels)?,
        };

        if user_vector.user_event_used_count == 0 {
            return Ok(vec!())
        }
        Ok(self.embedding_registry.embeddings
            .get(&query_index)
            .map(|index| index.search(user_vector.user_embedding.view_mut(), k))
            .unwrap_or(vec!()))
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
