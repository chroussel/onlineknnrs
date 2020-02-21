extern crate libc;
extern crate parquet;
#[macro_use] extern crate failure;
#[macro_use] extern crate log;
extern crate tempdir;

pub mod knnservice;
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
    pub fn to_native(&self) -> i32 {
        match self {
            Distance::Euclidean => { native::Distance_Euclidian }
            Distance::Angular => { native::Distance_Angular }
            Distance::InnerProduct => { native::Distance_InnerProduct }
        }
    }
}