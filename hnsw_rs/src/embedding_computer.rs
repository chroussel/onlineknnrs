use ndarray::Array1;
use crate::knnindex::EmbeddingRegistry;
use crate::KnnError;

pub struct EmbeddingResult {
    pub user_embedding: Array1<f32>,
    pub user_event_used_count: usize
}

pub struct UserEvent {
    pub index: i32,
    pub label: i64,
    pub timestamp: u64,
    pub event_type: i32
}

pub trait UserEmbeddingComputer: Sync + Send {
    fn compute_user_vector(&self, registry: &EmbeddingRegistry, user_events: &[UserEvent]) -> Result<EmbeddingResult, KnnError>;
}

#[derive(Default)]
pub struct AverageComputer {}

impl UserEmbeddingComputer for AverageComputer {
    fn compute_user_vector(&self, registry: &EmbeddingRegistry, user_events: &[UserEvent]) -> Result<EmbeddingResult, KnnError> {
        let mut count = 0;
        let mut user_vector = Array1::<f32>::zeros(registry.dim);

        for user_event in user_events {
            if let Some(data_vector) = registry.fetch_item(user_event.index, user_event.label) {
                count += 1;
                user_vector += &data_vector
            }
        }
        if count != 0 {
            user_vector /= count as f32;
        }

        Ok(EmbeddingResult {
            user_embedding: user_vector,
            user_event_used_count: count
        })
    }
}