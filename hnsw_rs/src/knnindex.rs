use crate::native;
use failure::Error;
use std::path::Path;
use crate::knnservice::Distance;
use std::ffi::CString;
use ndarray::{ArrayView, ArrayView1, Array1, ArrayViewMut1};
use std::fmt::Display;

pub struct KnnIndex {
    index: native::rust_hnsw_index_t,
    dim: i32,
}

impl KnnIndex {
    pub fn new(distance: Distance, dim: i32, ef_search: usize, maxSize: usize) -> Result<KnnIndex, Error> {
        const ef_construction: usize= 50;
        const M: usize= 50;
        const seed: usize= 42;
        let index = unsafe { native::create_index(distance.to_native(), dim) };
        unsafe {native::init_new_index(index, maxSize, M, ef_construction, seed)};
        unsafe { native::set_ef(index, ef_search) }
        Ok(KnnIndex { index, dim })
    }

    pub fn load<P: AsRef<Path> + Display>(distance: Distance, dim: i32, ef_search: usize, path_to_index: P) -> Result<KnnIndex, Error> {
        let index = unsafe { native::create_index(distance.to_native(), dim) };
        let path_str = path_to_index.as_ref().as_os_str().to_str().ok_or(format_err!("{} is invalid", path_to_index))?;
        let cs = CString::new(path_str).unwrap();
        unsafe { native::load_index(index, cs.as_ptr()) };
        Ok(KnnIndex { index, dim })
    }

    pub fn size(&self) -> usize {
        unsafe { native::cur_element_count(self.index) }
    }

    pub fn get_item(&self, label: u64) -> Option<ArrayViewMut1<f32>> {
        let pointer = unsafe { native::get_item(self.index, label as usize) };
        if pointer.is_null() {
            None
        } else {
            Some(unsafe{ArrayViewMut1::from_shape_ptr(self.dim as usize, pointer)})
        }
    }

    pub fn add_item(&self, label: u64, mut embedding: Array1<f32>) {
        unsafe { native::add_item(self.index, embedding.as_mut_ptr(), label as usize) }
    }

    pub fn query(&self, mut embedding: ArrayViewMut1<f32>, k: usize) -> Vec<(u64, f32)> {
        let mut items = vec!();
        items.reserve(k);
        let mut distances = vec!();
        distances.reserve(k);
        let nb_result = unsafe { native::query(self.index, embedding.as_mut_ptr(), items.as_mut_ptr(), distances.as_mut_ptr(), k) };
        unsafe {items.set_len(nb_result)};
        unsafe {distances.set_len(nb_result)};
        items.into_iter().map(|u| u as u64).zip(distances).collect()
    }
}

impl Drop for KnnIndex {
    fn drop(&mut self) {
        unsafe { native::destroy(self.index) };
    }
}

#[cfg(test)]
mod tests {
    use crate::knnindex::*;
    use crate::knnservice::Distance;
    use ndarray::*;

    #[test]
    fn create_and_query() {
        let a = KnnIndex::new(Distance::Euclidean, 3, 10, 100).unwrap();

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
