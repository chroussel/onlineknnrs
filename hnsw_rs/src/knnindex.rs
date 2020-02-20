use std::collections::HashMap;
use ndarray::*;
use crate::hnswindex::HnswIndex;
use std::collections::BinaryHeap;
use failure::_core::cmp::Ordering;

pub struct KnnIndex {
    hnsw_indexes: Vec<HnswIndex>,
    extra_items: HashMap<i64, Array1<f32>>
}

struct IndexResult(i64, f32);

impl Ord for IndexResult {
    fn cmp(&self, other: &Self) -> Ordering {
        let r = self.1.cmp(&other.1);
        if r == Ordering::Equal {
            self.0.cmp(&other.0)
        } else {
            r
        }
    }
}

impl KnnIndex {
    pub fn new() -> KnnIndex {
        KnnIndex {
            hnsw_indexes: vec!(),
            extra_items: HashMap::new()
        }
    }
    pub fn get_item(&self, label: i64) -> Option<ArrayView1<f32>> {
        self.extra_items.get(&label)
            .map(|e| e.view())
            .or_else(|| self.get_item_from_indices(label))
    }

    pub fn get_item_from_indices(&self, label: i64) -> Option<ArrayView1<f32>> {
        self.hnsw_indexes.iter().find_map(|index| index.get_item(label))
    }

    pub fn search(&self, embedding: ArrayViewMut1<f32>, nb_result: usize) -> Vec<(i64, f32)> {
        let mut init_heap = BinaryHeap::with_capacity(nb_result);
        self.hnsw_indexes.iter()
            .map(|index| index.query(embedding, nb_result))
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

pub struct EmbeddingRegistry {
    dim: i32,
    pub embeddings: HashMap<i32, KnnIndex>
}

impl EmbeddingRegistry {
    pub fn new(dim: i32) -> EmbeddingRegistry {
        EmbeddingRegistry {
            dim,
            embeddings:HashMap::new()
        }
    }

    pub fn fetch_item(&self, index_id: i32, label: i64) -> Option<ArrayView1<f32>> {
        self.embeddings.get(&index_id)
            .and_then(|index| index.get_item(label))
    }

    pub fn has_item(&self, index_id: i32, label: i64) -> bool {
        self.fetch_item(index, label).is_some()
    }
}

pub struct Model {
    half_life: f32,
    nb_last_days: i32,
    nb_last_events: i32,
    return_details: bool
}

