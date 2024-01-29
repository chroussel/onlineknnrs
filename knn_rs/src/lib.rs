use std::str::FromStr;

use crate::knnservice::Model;
use tensorflow::Status;
use thiserror::Error;

pub mod embedding_computer;
pub mod knn_tf;
pub mod knncountry;
pub mod knnindex;
pub mod knnservice;
pub mod loader;
pub mod productindex;
pub mod wrappedindex;

#[derive(Error, Debug)]
pub enum KnnError {
    #[error("Invalid path.")]
    InvalidPath,
    #[error("Index for id {0} is not found.")]
    IndexNotFound(i32),
    #[error("No vector has been found.")]
    NoVectorFound,
    #[error("An unknown error has occurred.")]
    UnknownError,
    #[error("Indexes are not loaded")]
    IndexNotLoaded,
    #[error("Can't parse {0} to distance")]
    UnknownDistance(String),
    #[error("Can't find model {0}")]
    ModelNotFound(String),
    #[error("No model specified")]
    ModelMissing,
    #[error("IoError {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization Error {0}")]
    SerdeError(#[from] serde_json::Error),
    #[error("Error in Faiss {0}")]
    FaissError(#[from] faiss::error::Error),
    #[error("Error in TF model {0}")]
    TFError(String),
    #[error("Not country {0} can be found to insert Model. Please load the country first")]
    CountryNotFoundWhileLoadingModel(String),
}

impl From<tensorflow::Status> for KnnError {
    fn from(status: Status) -> Self {
        KnnError::TFError(format!("{:?}", status))
    }
}

#[derive(Copy, Clone)]
pub struct IndexConfig {
    distance: Distance,
    dim: usize,
    ef_search: usize,
}

impl IndexConfig {
    pub fn new(distance: Distance, dim: usize, ef_search: usize) -> IndexConfig {
        IndexConfig {
            distance,
            dim,
            ef_search,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Angular,
    InnerProduct,
}

impl FromStr for Distance {
    type Err = KnnError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_ref() {
            "euclidean" => Ok(Distance::Euclidean),
            "angular" => Ok(Distance::Angular),
            "dotproduct" => Ok(Distance::InnerProduct),
            _ => Err(KnnError::UnknownDistance(value.to_string())),
        }
    }
}
