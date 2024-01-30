use crate::embedding_computer::{
    AverageComputer, EmbeddingResult, UserEmbeddingComputer, UserEvent,
};
use crate::knn_tf::KnnTf;
use crate::knnindex::EmbeddingRegistry;
use crate::loader::Loader;
use crate::productindex::ProductIndex;
use crate::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use self::productindex::IndexResult;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub name: String,
    pub model_path: Option<PathBuf>,
    pub model_type: ModelType,
    pub is_default: bool,
    pub version: Option<String>,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Deserialize)]
pub enum ModelType {
    Average,
    Tensorflow,
    XLA,
}

impl FromStr for ModelType {
    type Err = KnnError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "average" | "avg" => Ok(ModelType::Average),
            "tf" | "tensorflow" => Ok(ModelType::Tensorflow),
            "xla" => Ok(ModelType::XLA),
            _ => Err(KnnError::ModelNotFound(s.to_string())),
        }
    }
}

pub struct KnnService {
    embedding_registry: Option<EmbeddingRegistry>,
    default_model: Option<String>,
    models: HashMap<String, Box<dyn UserEmbeddingComputer>>,
}

impl KnnService {
    pub fn new() -> Self {
        KnnService {
            embedding_registry: None,
            default_model: None,
            models: HashMap::new(),
        }
    }

    pub fn list_labels(&self, partner_id: i32) -> Result<Vec<i64>, KnnError> {
        if let Some(emr) = self.embedding_registry.as_ref() {
            return emr.list_labels(partner_id);
        } else {
            return Err(KnnError::IndexNotLoaded);
        }
    }

    pub fn get_item(&self, partner_id: i32, label: i64) -> Result<Option<Vec<f32>>, KnnError> {
        if let Some(emr) = self.embedding_registry.as_ref() {
            return emr.fetch_item(partner_id, label);
        } else {
            return Err(KnnError::IndexNotLoaded);
        }
    }

    pub fn load_index<P: AsRef<Path>>(&mut self, indices_path: P) -> Result<(), KnnError> {
        info!(
            "KnnService: Starting load from {}",
            indices_path.as_ref().display(),
        );
        let map = Loader::load_index_folder(indices_path)?;
        if let Some((_, i)) = map.iter().next() {
            let dim = i.dimension();
            let registry = EmbeddingRegistry::new(dim, map);
            self.embedding_registry.replace(registry);
        }

        info!("KnnService: Load done");
        Ok(())
    }

    pub fn load_model<P: AsRef<Path>>(
        &mut self,
        model: Model,
        tf_model: Option<P>,
    ) -> Result<(), KnnError> {
        info!("KnnService: Starting model load of {}", model.name);
        let computer: Box<dyn UserEmbeddingComputer> = match model.model_type {
            ModelType::Average => Box::new(AverageComputer::default()),
            ModelType::Tensorflow => {
                if let Some(mp) = tf_model {
                    Box::new(KnnTf::load_model(mp)?)
                } else {
                    return Err(KnnError::InvalidPath);
                }
            }
            ModelType::XLA => unimplemented!(),
        };
        self.models.insert(model.name.clone(), computer);
        if model.is_default {
            self.default_model = Some(model.name.clone())
        }
        info!("KnnService: Model load done");
        Ok(())
    }

    fn compute_user_vector(
        &self,
        model: Option<String>,
        user_events: &[UserEvent],
    ) -> Result<EmbeddingResult, KnnError> {
        let model_name = model.or(self.default_model.clone());
        if let Some(model_name) = model_name {
            if let Some(emr) = self.embedding_registry.as_ref() {
                self.models
                    .get(&model_name)
                    .ok_or(KnnError::ModelNotFound(model_name))
                    .and_then(|m| m.compute_user_vector(emr, user_events))
                    .map_err(From::from)
            } else {
                return Err(KnnError::IndexNotLoaded);
            }
        } else {
            return Err(KnnError::ModelMissing);
        }
    }

    pub fn get_closest_items(
        &self,
        user_events: &[UserEvent],
        query_index: i32,
        k: usize,
        model: Option<String>,
    ) -> Result<Vec<IndexResult>, KnnError> {
        let user_vector = self.compute_user_vector(model, user_events)?;

        if user_vector.user_event_used_count == 0 {
            return Ok(vec![]);
        }

        if let Some(emr) = self.embedding_registry.as_ref() {
            if let Some(index) = emr.embeddings.get(&query_index) {
                return index.search(&user_vector.user_embedding, k);
            } else {
                return Ok(vec![]);
            }
        } else {
            return Err(KnnError::IndexNotLoaded);
        }
    }
}
