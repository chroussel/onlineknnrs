use crate::native;
use std::path::Path;
use std::ffi::CString;
use std::collections::HashMap;
use ndarray::Array1;

#[derive(Debug)]
pub enum Error {
    InvalidPath,
    IndexNotFound(i32),
    NoVectorFound,
}

pub enum Distance {
    Euclidean,
    Angular,
    InnerProduct,
}

pub enum Model {
    Average
}

impl Distance {
    fn to_native(&self) -> i32 {
        match self {
            Distance::Euclidean => { return native::Distance_Euclidian; }
            Distance::Angular => { return native::Distance_Angular; }
            Distance::InnerProduct => { return native::Distance_InnerProduct; }
        }
    }
}

pub struct KnnService {
    distance: Distance,
    dim: i32,
    ef_search: usize,
    indices_by_id: HashMap<i32, native::rust_hnsw_index_t>,
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

    pub fn load_index<P: AsRef<Path>>(&mut self, index_id: i32, path_to_index: P) -> Result<(), Error> {
        let index = unsafe { native::create_index(self.distance.to_native(), self.dim) };
        let path_str = path_to_index.as_ref().as_os_str().to_str().ok_or(Error::InvalidPath)?;
        let cs = CString::new(path_str).unwrap();
        unsafe { native::load_index(index, cs.as_ptr()) };
        unsafe { native::set_ef(index, self.ef_search) }
        self.indices_by_id.insert(index_id, index);
        Ok(())
    }

    fn compute_user_vector(&self, labels: &Vec<(i32, i64)>) -> Result<Array1<f32>, Error> {
        let mut count = 0;
        let mut user_vector = Array1::<f32>::zeros(self.dim as usize);
        let mut data_vector = Array1::<f32>::zeros(self.dim as usize);

        for (index_id, label) in labels {
            self.indices_by_id.get(&index_id).into_iter().for_each(
                |index| {
                    let found = unsafe { native::get_data_pointer_by_label(*index, *label as usize, data_vector.as_mut_ptr()) };
                    if found {
                        count += 1;
                        user_vector += &data_vector;
                    }
                }
            );
        }
        if count == 0 {
            return Err(Error::NoVectorFound);
        }
        user_vector /= (count as f32);
        return Ok(user_vector);
    }

    pub fn get_closest_items(&self, labels: &Vec<(i32, i64)>, query_index: i32, k: usize, model: Model) -> Result<Vec<(i64, f32)>, Error> {
        let mut user_vector = match model {
            Model::Average => self.compute_user_vector(&labels)?,
        };

        let mut items = vec!();
        items.reserve(k);

        let mut distances = vec!();
        distances.reserve(k);

        self.indices_by_id
            .get(&query_index)
            .and_then(|index| {
                let nb_result = unsafe { native::query(*index, user_vector.as_mut_ptr(), items.as_mut_ptr(), distances.as_mut_ptr(), k) };
                Some(items.into_iter().take(nb_result).map(|i| i as i64).zip(distances).collect())
            }).ok_or(Error::IndexNotFound(query_index))
    }
}

impl Drop for KnnService {
    fn drop(&mut self) {
        self.indices_by_id.iter().for_each(|(_, index)| {
            unsafe { native::destroy(*index) }
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::knnservice::*;
    use test::Bencher;

    #[test]
    fn simple_test() {
        let a = KnnService::new(Distance::Euclidean, 100, 10);

    }
}
