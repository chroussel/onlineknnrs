use std::path::Path;
use std::collections::HashMap;
use ndarray::{Array1, ArrayViewMut1};
use crate::knnindex::*;
use crate::error::KnnError;

pub enum Model {
    Average
}

pub struct KnnService {
    distance: Distance,
    dim: i32,
    ef_search: usize,
    indices_by_id: HashMap<i32, KnnIndex>,
}

impl KnnService {
    pub fn new(distance: Distance, dim: i32, ef_search: usize) -> Self {
        KnnService {
            distance,
            dim,
            ef_search,
            indices_by_id: HashMap::new(),
        }
    }

    pub fn load_index<P: AsRef<Path>>(&mut self, index_id: i32, path_to_index: P) -> Result<(), KnnError> {
        let index = KnnIndex::load(self.distance, self.dim, self.ef_search, path_to_index)?;
        self.indices_by_id.insert(index_id, index);
        Ok(())
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
    use crate::knnindex::Distance;

    #[test]
    fn simple_test() {
        let a = KnnService::new(Distance::Euclidean, 100, 10);

    }
}
