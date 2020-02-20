extern crate libc;
extern crate parquet;
#[macro_use] extern crate failure;

pub mod error;
pub mod knnservice;
pub mod hnswindex;
pub mod knnindex;
pub mod loader;
#[allow(dead_code)]
#[allow(non_upper_case_globals)]
mod native;

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