use crate::{native, Distance, IndexConfig, KnnError};
use std::path::Path;
use std::ffi::CString;
use ndarray::*;
use std::collections::HashMap;

pub struct HnswIndex {
    index: native::RustHnswIndexT,
    config: IndexConfig
}

impl HnswIndex {
    pub fn new(config: IndexConfig, max_size: usize) -> Result<HnswIndex, KnnError> {
        const EF_CONSTRUCTION: usize= 50;
        const M: usize= 50;
        const SEED: usize= 42;
        let index = unsafe { native::create_index(config.distance.to_native(), config.dim as i32) };
        unsafe {native::init_new_index(index, max_size, M, EF_CONSTRUCTION, SEED)};
        unsafe { native::set_ef(index, config.ef_search) }
        Ok(HnswIndex { index, config })
    }

    pub fn load<P: AsRef<Path>>(config: IndexConfig, path_to_index: P) -> Result<HnswIndex, KnnError> {
        let index = unsafe { native::create_index(config.distance.to_native(), config.dim as i32) };
        let path_str = path_to_index.as_ref().as_os_str().to_str().ok_or(KnnError::InvalidPath)?;
        let cs = CString::new(path_str).unwrap();
        unsafe { native::load_index(index, cs.as_ptr()) };
        unsafe { native::set_ef(index, config.ef_search) }
        Ok(HnswIndex { index, config })
    }

    pub fn size(&self) -> usize {
        unsafe { native::cur_element_count(self.index) }
    }

    pub fn get_item(&self, label: i64) -> Option<ArrayView1<f32>> {
        let pointer = unsafe { native::get_item(self.index, label as usize) };
        if pointer.is_null() {
            None
        } else {
            Some(unsafe{ArrayView1::from_shape_ptr(self.config.dim as usize, pointer)})
        }
    }

    pub fn add_item(&self, label: i64, mut embedding: Array1<f32>) {
        unsafe { native::add_item(self.index, embedding.as_mut_ptr(), label as usize) }
    }

    pub fn query(&self, embedding: &mut ArrayViewMut1<f32>, k: usize) -> Vec<(i64, f32)> {
        let mut items = vec!();
        items.reserve(k);
        let mut distances = vec!();
        distances.reserve(k);
        let nb_result = unsafe { native::query(self.index, embedding.as_mut_ptr(), items.as_mut_ptr(), distances.as_mut_ptr(), k) };
        unsafe {items.set_len(nb_result)};
        unsafe {distances.set_len(nb_result)};
        items.into_iter().map(|u| u as i64).zip(distances).collect()
    }
}

impl Drop for HnswIndex {
    fn drop(&mut self) {
        unsafe { native::destroy(self.index) };
    }
}

#[cfg(test)]
mod tests {
    use ndarray::*;
    use crate::*;
    use crate::hnswindex::HnswIndex;

    #[test]
    fn create_and_query() {
        let config = IndexConfig::new(Distance::Euclidean, 3, 10, 100);
        let a = HnswIndex::new(config, 100).unwrap();

        for i in 0..50 {
            let mut e = arr1(&[i as f32, (i+1) as f32, (i+2) as f32]);
            a.add_item(i, e);
        }

        let mut query = arr1(&[20_f32, 21.0, 22.0]);
        let res = a.query(query.view_mut(), 5);
        assert_eq!(res.len(), 5);
        dbg!(res);
    }
}
