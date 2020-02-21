use std::path::{Path, PathBuf};
use std::collections::HashMap;
use ndarray::{Array1, ArrayViewMut1, ArrayView1};
use crate::hnswindex::*;
use crate::*;
use crate::knnindex::{EmbeddingRegistry, KnnIndex};
use crate::loader::Loader;
use failure::Error;

pub enum Model {
    Average
}

pub struct KnnService {
    config: IndexConfig,
    embedding_registry: EmbeddingRegistry
}

impl KnnService {
    pub fn new(config: IndexConfig) -> Self {
        KnnService {
            config,
            embedding_registry: EmbeddingRegistry::new()
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

    pub  fn compute_user_vector(&self, labels: &[(i32, i64)]) -> Result<Array1<f32>, Error> {
        let mut count = 0;
        let mut user_vector = Array1::<f32>::zeros(self.config.dim);

        for &(index_id, label) in labels {
            if let Some(data_vector) = self.embedding_registry.fetch_item(index_id, label) {
                count += 1;
                user_vector += &data_vector
            }
        }
        if count == 0 {
            return Err(KnnError::NoVectorFound.into());
        }
        user_vector /= count as f32;
        Ok(user_vector)
    }

    pub fn get_closest_items(&self, labels: &[(i32, i64)], query_index: i32, k: usize, model: Model) -> Result<Vec<(i64, f32)>, Error> {
        let mut user_vector = match model {
            Model::Average => self.compute_user_vector(labels)?,
        };

        self.embedding_registry.embeddings
            .get(&query_index)
            .and_then(|index| Some(index.search(user_vector.view_mut(), k)))
            .ok_or(KnnError::IndexNotFound(query_index).into())
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
