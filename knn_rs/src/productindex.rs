use crate::KnnError;

pub trait ProductIndex {
    fn count(&self) -> usize;
    fn dimension(&self) -> usize;
    fn list_labels(&self) -> Result<Vec<i64>, KnnError>;
    fn get_item(&self, id: i64) -> Result<Option<Vec<f32>>, KnnError>;
    fn search(&self, embedding: &[f32], output: usize) -> Result<Vec<IndexResult>, KnnError>;
}

#[derive(Debug, PartialEq)]
pub struct IndexResult(pub i64, pub f32);

impl PartialOrd for IndexResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

impl Eq for IndexResult {}

impl Ord for IndexResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.1.total_cmp(&other.1)
    }
}
