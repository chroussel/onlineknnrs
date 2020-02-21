#[macro_use] extern crate log;
extern crate env_logger;

use tonic::{transport::Server, Request, Response, Status};
use knn::{*, knn_server::*};
mod knn {
    tonic::include_proto!("knn.api");
}
use env_logger::Env;


#[derive(Default)]
pub struct KnnController {}

#[tonic::async_trait]
impl Knn for KnnController {
    async fn search(
        &self,
        _request: Request<KnnRequest>,
    ) -> Result<Response<KnnResponse>, Status> {
        unimplemented!()
    }
    async fn multi_search(
        &self,
        _request: Request<KnnRequest>,
    ) -> Result<Response<KnnResponse>, Status> {
        unimplemented!()
    }
    async fn get_available_countries(
        &self,
        _request: Request<()>,
    ) -> Result<Response<AvailableCountriesResponse>, Status> {
        unimplemented!()
    }
    async fn get_indices_for_country(
        &self,
        _request: Request<IndicesRequest>,
    ) -> Result<Response<IndicesResponse>, Status> {
        unimplemented!()
    }
    async fn get_indexed_products(
        &self,
        _request: Request<IndexedProductsRequest>,
    ) -> Result<Response<IndexedProductsResponse>, Status> {
        unimplemented!()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>>{
    env_logger::from_env(Env::default().default_filter_or("info")).init();
    let addr = "[::1]:8080".parse().unwrap();
    info!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(KnnController::default()))
        .serve(addr)
        .await?;
    Ok(())
}
