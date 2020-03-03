use std::collections::HashMap;
use ndarray::*;
use crate::hnswindex::HnswIndex;
use std::collections::BinaryHeap;
use failure::_core::cmp::Ordering;
use std::path::Path;
use crate::*;
use failure::Error;

pub struct KnnIndex {
    config: IndexConfig,
    hnsw_indexes: Vec<HnswIndex>,
    extra_items: HashMap<i64, Array1<f32>>
}

struct IndexResult(i64, f32);

impl Eq for IndexResult {}

impl PartialEq for IndexResult {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1.eq(&other.1)
    }
}

impl Ord for IndexResult {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.partial_cmp(other) {
            None => Ordering::Equal,
            Some(r) => r,
        }
    }
}

impl PartialOrd for IndexResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let r = self.1.partial_cmp(&other.1);
        if let Some(Ordering::Equal) = r {
            self.0.partial_cmp(&other.0)
        } else {
            r
        }
    }
}

impl KnnIndex {
    pub fn new(config: IndexConfig) -> KnnIndex {
        KnnIndex {
            config,
            hnsw_indexes: vec!(),
            extra_items: HashMap::new()
        }
    }

    pub fn add_index_from_path<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Error>{
        let wrapper = HnswIndex::load(self.config, path)?;
        self.hnsw_indexes.push(wrapper);
        Ok(())
    }

    pub fn add_extra_item(&mut self, label: i64, embedding: Array1<f32>) {
        self.extra_items.insert(label, embedding);
    }

    pub fn get_item(&self, label: i64) -> Option<ArrayView1<f32>> {
        self.extra_items.get(&label)
            .map(|e| e.view())
            .or_else(|| self.get_item_from_indices(label))
    }

    pub fn get_item_from_indices(&self, label: i64) -> Option<ArrayView1<f32>> {
        self.hnsw_indexes.iter().find_map(|index| index.get_item(label))
    }

    pub fn search(&self, mut embedding: ArrayViewMut1<f32>, nb_result: usize) -> Vec<(i64, f32)> {
        let init_heap = BinaryHeap::with_capacity(nb_result);
        self.hnsw_indexes.iter()
            .map(|index| index.query(&mut embedding, nb_result))
            .fold(init_heap, |mut heap, item| {
                item.into_iter().for_each(|(label, distance)| {
                    if heap.len() < nb_result {
                        heap.push(IndexResult(label, distance));
                    } else {
                        heap.pop();
                    }
                });
                heap
            })
            .into_iter()
            .map(|IndexResult(label, distance)| (label, distance))
            .collect()
    }
}

#[derive(Default)]
pub struct EmbeddingRegistry {
    pub dim: usize,
    zero: Array1<f32>,
    pub embeddings: HashMap<i32, KnnIndex>
}

impl EmbeddingRegistry {
    pub fn new(dim: usize) -> EmbeddingRegistry {
        EmbeddingRegistry {
            dim,
            zero: Array1::<f32>::zeros(dim),
            embeddings:HashMap::new()
        }
    }

    pub fn zero(&self) -> ArrayView1<f32> {
        self.zero.view()
    }

    pub fn fetch_item(&self, index_id: i32, label: i64) -> Option<ArrayView1<f32>> {
        self.embeddings.get(&index_id)
            .and_then(|index| index.get_item(label))
    }

    pub fn has_item(&self, index_id: i32, label: i64) -> bool {
        self.fetch_item(index_id, label).is_some()
    }
}

