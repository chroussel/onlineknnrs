extern crate libc;
extern crate parquet;
#[macro_use] extern crate failure;
#[macro_use] extern crate log;
extern crate tempdir;
extern crate tensorflow;

use failure::_core::str::FromStr;
use crate::knnservice::Model;
use tensorflow::Status;

pub mod embedding_computer;
pub mod knnservice;
pub mod knn_tf;
pub mod knncountry;
pub mod hnswindex;
pub mod knnindex;
pub mod loader;
#[allow(dead_code)]
#[allow(non_upper_case_globals)]
mod native;

#[derive(Fail, Debug)]
pub enum KnnError {
    #[fail(display = "Invalid path.")]
    InvalidPath,
    #[fail(display = "Index for id {} is not found.", _0)]
    IndexNotFound(i32),
    #[fail(display = "No vector has been found.")]
    NoVectorFound,
    #[fail(display = "An unknown error has occurred.")]
    UnknownError,
    #[fail(display = "Can't parse {} to distance", _0)]
    UnknownDistance(String),
    #[fail(display = "Can't find model {}", _0)]
    ModelNotFound(Model),
    #[fail(display = "Error in TF model {}", _0)]
    TFError(String),
    #[fail(display = "Not country {} can be found to insert Model. Please load the country first", _0)]
    CountryNotFoundWhileLoadingModel(String)
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
    ef_search: usize
}

impl IndexConfig {
    pub fn new(distance: Distance, dim: usize, ef_search: usize) -> IndexConfig {
        IndexConfig {
            distance,
            dim,
            ef_search
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Angular,
    InnerProduct,
}

impl Distance {
    pub fn to_native(self) -> i32 {
        match self {
            Distance::Euclidean => { native::Distance_Euclidian }
            Distance::Angular => { native::Distance_Angular }
            Distance::InnerProduct => { native::Distance_InnerProduct }
        }
    }
}

impl FromStr for Distance {
    type Err = KnnError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_ref() {
            "euclidean" => Ok(Distance::Euclidean),
            "angular" => Ok(Distance::Angular),
            "dotproduct" => Ok(Distance::InnerProduct),
            _ => Err(KnnError::UnknownDistance(value.to_string()))
        }
    }
}