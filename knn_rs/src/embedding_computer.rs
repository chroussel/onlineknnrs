use crate::knnindex::EmbeddingRegistry;
use crate::KnnError;
use ndarray::{Array1, ArrayView1};

pub struct EmbeddingResult {
    pub user_embedding: Vec<f32>,
    pub user_event_used_count: usize,
}

#[derive(Clone)]
pub struct UserEvent {
    pub index: i32,
    pub label: i64,
    pub timestamp: u64,
    pub event_type: i32,
}

pub trait UserEmbeddingComputer: Sync + Send {
    fn compute_user_vector(
        &self,
        registry: &EmbeddingRegistry,
        user_events: &[UserEvent],
    ) -> Result<EmbeddingResult, KnnError>;
}

#[derive(Default)]
pub struct AverageComputer {}

impl UserEmbeddingComputer for AverageComputer {
    fn compute_user_vector(
        &self,
        registry: &EmbeddingRegistry,
        user_events: &[UserEvent],
    ) -> Result<EmbeddingResult, KnnError> {
        let mut count = 0;
        let mut user_vector = Array1::<f32>::zeros(registry.dim);

        for user_event in user_events {
            if let Some(data_vector) = registry.fetch_item(user_event.index, user_event.label)? {
                count += 1;
                let view = ArrayView1::from(data_vector.as_slice());
                user_vector += &view;
            }
        }
        if count != 0 {
            user_vector /= count as f32;
        }

        Ok(EmbeddingResult {
            user_embedding: user_vector.to_vec(),
            user_event_used_count: count,
        })
    }
}
