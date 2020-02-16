extern crate libc;
#[macro_use] extern crate failure;

pub mod error;
pub mod knnservice;
pub mod knnindex;
#[allow(dead_code)]
#[allow(non_upper_case_globals)]
mod native;