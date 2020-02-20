use std::path::{Path, PathBuf};
use std::collections::HashMap;
use ndarray::{Array1, ArrayViewMut1};
use crate::hnswindex::*;
use crate::error::KnnError;
use crate::Distance;
use crate::knnindex::{EmbeddingRegistry, KnnIndex};
use crate::loader::Loader;

pub enum Model {
    Average
}

pub struct KnnService {
    distance: Distance,
    dim: i32,
    ef_search: usize,
    embedding_registry: EmbeddingRegistry
}

impl KnnService {
    pub fn new(distance: Distance, dim: i32, ef_search: usize) -> Self {
        KnnService {
            distance,
            dim,
            ef_search,
            embedding_registry: EmbeddingRegistry::new(dim)
        }
    }

    pub fn load<P: AsRef<Path>>(&mut self, indices_path: P, extra_item_path: P) -> Result<(), KnnError> {
        info!("KnnService: Starting load from {} and {}", indices_path.as_ref(), extra_item_path.as_ref());
        Loader::load_index_folder(indices_path, self.add_index)?;
        Loader::load_extra_item_folder(extra_item_path, self.add_extra_item)?;
        info!("KnnService: Load done");
        Ok(())
    }

    fn add_index(&mut self, index_id: i32, path: PathBuf) -> Result<(), KnnError> {

    }

    fn add_extra_item(&mut self, index_id: i32, label: i64, embedding: Vec<f32> ) -> Result<(), KnnError> {

    }

    fn get_item(&self, index: i32, label: i64) -> Option<ArrayViewMut1<f32>> {
        self.indices_by_id.get(&index)
            .and_then(|index| index.get_item(label))
    }

    fn compute_user_vector(&self, labels: &[(i32, i64)]) -> Result<Array1<f32>, KnnError> {
        let mut count = 0;
        let mut user_vector = Array1::<f32>::zeros(self.dim as usize);

        for &(index_id, label) in labels {
            if let Some(data_vector) = self.get_item(index_id, label) {
                count += 1;
                user_vector += &data_vector
            }
        }
        if count == 0 {
            return Err(KnnError::NoVectorFound);
        }
        user_vector /= count as f32;
        Ok(user_vector)
    }

    pub fn get_closest_items(&self, labels: &[(i32, i64)], query_index: i32, k: usize, model: Model) -> Result<Vec<(i64, f32)>, KnnError> {
        let mut user_vector = match model {
            Model::Average => self.compute_user_vector(labels)?,
        };

        self.indices_by_id
            .get(&query_index)
            .and_then(|index| Some(index.query(user_vector.view_mut(), k)))
            .ok_or(KnnError::IndexNotFound(query_index))
    }
}

#[cfg(test)]
mod tests {
    use crate::knnservice::*;
    use crate::hnswindex::Distance;
    use crate::Distance;

    #[test]
    fn simple_test() {
        let a = KnnService::new(Distance::Euclidean, 100, 10);

    }
}
