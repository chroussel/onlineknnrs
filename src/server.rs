use tonic::{transport::Server, Request, Response, Status};
use knn::{*, knn_server::*};
mod knn {
    tonic::include_proto!("knn.api");
}

#[derive(Default)]
pub struct KnnController {}

#[tonic::async_trait]
impl Knn for KnnController {
    async fn search(
        &self,
        request: Request<KnnRequest>,
    ) -> Result<Response<KnnResponse>, Status> {
        unimplemented!()
    }
    async fn multi_search(
        &self,
        request: Request<KnnRequest>,
    ) -> Result<Response<KnnResponse>, Status> {
        unimplemented!()
    }
    async fn get_available_countries(
        &self,
        request: Request<()>,
    ) -> Result<Response<AvailableCountriesResponse>, Status> {
        unimplemented!()
    }
    async fn get_indices_for_country(
        &self,
        request: Request<IndicesRequest>,
    ) -> Result<Response<IndicesResponse>, Status> {
        unimplemented!()
    }
    async fn get_indexed_products(
        &self,
        request: Request<IndexedProductsRequest>,
    ) -> Result<Response<IndexedProductsResponse>, Status> {
        unimplemented!()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>>{
    let addr = "[::1]:8080".parse().unwrap();
    println!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(KnnController::default()))
        .serve(addr)
        .await?;
    Ok(())
}
