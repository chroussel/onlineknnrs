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