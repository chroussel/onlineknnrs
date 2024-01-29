use parking_lot::RwLock;
use std::{collections::HashMap, sync::Arc};

use crate::{
    productindex::{IndexResult, ProductIndex},
    KnnError,
};

pub struct WrappedIndex {
    // Product to faiss id mapping
    mapping: HashMap<i64, faiss::Idx>,
    norm: Vec<f32>,
    index: Arc<RwLock<Box<dyn faiss::Index>>>,
}

impl WrappedIndex {
    pub fn new(
        index: Box<dyn faiss::Index>,
        mapping: HashMap<i64, faiss::Idx>,
        norm: Vec<f32>,
    ) -> WrappedIndex {
        WrappedIndex {
            index: Arc::new(RwLock::new(index)),
            mapping,
            norm,
        }
    }
}

impl ProductIndex for WrappedIndex {
    fn count(&self) -> usize {
        let rguard = self.index.read();
        rguard.ntotal() as usize
    }

    fn dimension(&self) -> usize {
        let rguard = self.index.read();
        rguard.d() as usize
    }

    fn get_item(&self, id: i64) -> Result<Option<Vec<f32>>, KnnError> {
        let inner_id = self.mapping.get(&id);
        match inner_id {
            Some(id) => {
                let rguard = self.index.read();
                Ok(Some(rguard.reconstruct(*id)?))
            }
            None => Ok(None),
        }
    }

    fn search(&self, embedding: &[f32], k: usize) -> Result<Vec<IndexResult>, KnnError> {
        let mut wguard = self.index.write();
        let r = wguard.search(embedding, k)?;
        let res = r
            .labels
            .into_iter()
            .map(|d| d.to_native())
            .zip(r.distances.into_iter())
            .map(|a| IndexResult(a.0, a.1))
            .collect();
        Ok(res)
    }
}
