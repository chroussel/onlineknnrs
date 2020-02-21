#[macro_use] extern crate log;
extern crate env_logger;
extern crate hnsw_rs;

use tonic::{transport::Server, Request, Response, Status};
use knn::{*, knn_server::*};

mod knn;

use env_logger::Env;
use hnsw_rs::knnservice::Model;
use failure::Error;
use std::path::{Path, PathBuf};
use clap::{App, Arg};
use hnsw_rs::knncountry::KnnByCountry;
use std::collections::HashSet;

pub struct KnnController {
    countries: HashSet<String>,
    knn_country:KnnByCountry
}
impl KnnController {
    fn new(countries: Vec<String>) -> KnnController {
        KnnController {
            countries: countries.into_iter().map(|c| c.to_uppercase()).collect(),
            knn_country: KnnByCountry::default()
        }
    }

    fn load<P>(&mut self, index_path: P, extra_item_path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        let indices_path = index_path.as_ref();
        let extra_item_path = extra_item_path.as_ref();
        self.countries.clone().into_iter().map(move |c| {
            info!("Loading country {}", c);
            let load_result = self.knn_country.load(&c,
                                                    indices_path,
                                                    extra_item_path);
            match &load_result {
                Ok(()) => info!("Done for {}", c),
                Err(e) => error!("error loading {}: {}", c, e.to_string())
            }
            load_result
        }).collect()
    }

    fn build_response(products: Vec<(i64, f32)>) -> KnnResponse{
        let mut response = KnnResponse::default();
        response.products = products.iter().map(|(label, score)| {
            knn::Product {
                product_id: *label,
                score: *score,
                dotproduct: 0f32,
                squared_l2_norm: 0f32
            }
        }).collect();
        response
    }
}
#[tonic::async_trait]
impl Knn for KnnController {
    async fn search(
        &self,
        request: Request<KnnRequest>,
    ) -> Result<Response<KnnResponse>, Status> {
        let request: KnnRequest = request.into_inner();
        debug!("Received request with country: {}", request.country);
        if let Some(knn_service) = self.knn_country.get_service(&request.country) {

            let events: Vec<(i32, i64)> = request.user_events.iter().map(|event| (event.partner_id, event.product_id)).collect();
            let result = knn_service.get_closest_items(
                &events,
                request.index_id,
                request.result_count as usize,
                Model::Average);

            match result {
                Ok(r) => {
                    let response = KnnController::build_response(r);
                    Ok(Response::new(response))
                },
                Err(error) => Err(Status::internal(error.to_string()))
            }
        } else {
            Err(Status::not_found(format!("country {} not available", request.country)))
        }
    }
    async fn multi_search(
        &self,
        _request: Request<KnnRequest>,
    ) -> Result<Response<KnnResponse>, Status> {
        Err(Status::unimplemented(""))
    }
    async fn get_available_countries(
        &self,
        _request: Request<()>,
    ) -> Result<Response<AvailableCountriesResponse>, Status> {
        let countries: Vec<CountryInfo> = self.knn_country.get_countries().into_iter().map(|c| {
            let mut country_info = CountryInfo::default();
            country_info.name = c;
            country_info
        }).collect();
        Ok(Response::new(
            AvailableCountriesResponse {
            countries
        }))
    }
    async fn get_indices_for_country(
        &self,
        _request: Request<IndicesRequest>,
    ) -> Result<Response<IndicesResponse>, Status> {
        Err(Status::unimplemented(""))
    }
    async fn get_indexed_products(
        &self,
        _request: Request<IndexedProductsRequest>,
    ) -> Result<Response<IndexedProductsResponse>, Status> {
        Err(Status::unimplemented(""))
    }
}

struct ResultArgs {
    port: u16,
    index_path: PathBuf,
    extra_item_path: PathBuf,
}

fn parse_args() -> Result<ResultArgs, Error> {
    let matches = App::new("Knn Serving")
        .arg(Arg::with_name("port").short("p").long("port").value_name("PORT").takes_value(true).required(true))
        .arg(Arg::with_name("index_path").short("i").long("index_path").value_name("PATH").takes_value(true).required(true))
        .arg(Arg::with_name("extra_path").short("e").long("extra_path").value_name("PATH").takes_value(true).required(true))
        .get_matches();

    let port: u16 = matches.value_of("port").unwrap().parse()?;
    let index_path: PathBuf = PathBuf::from(matches.value_of("index_path").unwrap());
    let extra_item_path: PathBuf = PathBuf::from(matches.value_of("extra_path").unwrap());

    Ok(ResultArgs {
        port,
        index_path,
        extra_item_path
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>>{
    env_logger::from_env(Env::default().default_filter_or("info")).init();
    let result_args = parse_args()?;
    let addr = format!("[::1]:{}", result_args.port).parse().unwrap();
    info!("initializing server");
    let mut controller = KnnController::new(vec!(String::from("FR")));
    controller.load(&result_args.index_path, &result_args.extra_item_path)?;

    info!("Starting server on {}", addr);
    Server::builder()
        .add_service(KnnServer::new(controller))
        .serve(addr)
        .await?;
    Ok(())
}
