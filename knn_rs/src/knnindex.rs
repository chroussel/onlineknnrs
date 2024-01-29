use crate::*;
use serde::Deserialize;
use std::collections::BinaryHeap;
use std::collections::HashMap;

use self::productindex::IndexResult;
use self::productindex::ProductIndex;
use self::wrappedindex::WrappedIndex;

pub struct KnnIndex {
    indices: Vec<WrappedIndex>,
    extra_items: Vec<WrappedIndex>,
}

#[derive(Deserialize)]
pub struct Metadata {
    pub(crate) partner_id: i32,
    pub(crate) chunk_id: i32,
    pub(crate) count: usize,
    pub(crate) country: String,
    pub(crate) index_param: String,
    pub(crate) is_recommendable: bool,
    pub(crate) metrics: String,
    pub(crate) metric: String,
    pub(crate) dimension: usize,
}

impl ProductIndex for KnnIndex {
    fn count(&self) -> usize {
        if let Some(i) = self.indices.first() {
            return i.count();
        }
        if let Some(i) = self.extra_items.first() {
            return i.count();
        }
        return 0;
    }

    fn dimension(&self) -> usize {
        if let Some(i) = self.indices.first() {
            return i.dimension();
        }
        if let Some(i) = self.extra_items.first() {
            return i.dimension();
        }
        return 0;
    }

    fn get_item(&self, label: i64) -> Result<Option<Vec<f32>>, KnnError> {
        for i in self.indices.iter() {
            if let Some(v) = i.get_item(label)? {
                return Ok(Some(v));
            }
        }
        for i in self.extra_items.iter() {
            if let Some(v) = i.get_item(label)? {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    fn search(&self, embedding: &[f32], nb_result: usize) -> Result<Vec<IndexResult>, KnnError> {
        let mut heap = BinaryHeap::with_capacity(nb_result);
        for index in self.indices.iter() {
            let res = index.search(embedding, nb_result)?;
            res.into_iter().for_each(|result| {
                if heap.len() >= nb_result {
                    heap.pop();
                }
                heap.push(result);
            });
        }
        Ok(heap.into_vec())
    }
}

impl KnnIndex {
    pub fn new() -> KnnIndex {
        KnnIndex {
            indices: vec![],
            extra_items: vec![],
        }
    }

    pub fn add_reco_index(&mut self, wi: WrappedIndex) {
        self.indices.push(wi);
    }

    pub fn add_non_reco_index(&mut self, wi: WrappedIndex) {
        self.extra_items.push(wi);
    }
}

#[derive(Default)]
pub struct EmbeddingRegistry {
    pub dim: usize,
    pub embeddings: HashMap<i32, KnnIndex>,
}

impl EmbeddingRegistry {
    pub fn new(dim: usize, embeddings: HashMap<i32, KnnIndex>) -> EmbeddingRegistry {
        EmbeddingRegistry { dim, embeddings }
    }

    pub fn fetch_item(&self, index_id: i32, label: i64) -> Result<Option<Vec<f32>>, KnnError> {
        if let Some(index) = self.embeddings.get(&index_id) {
            return index.get_item(label);
        }
        return Ok(None);
    }

    pub fn has_item(&self, index_id: i32, label: i64) -> Result<bool, KnnError> {
        self.fetch_item(index_id, label).map(|a| a.is_some())
    }
}
