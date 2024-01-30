use crate::knn::{knn_server::*, *};
use anyhow::Result;
use knn_rs::knncountry::{KnnByCountry, KnnConfig};
use knn_rs::{embedding_computer::UserEvent, productindex::IndexResult};
use metrics::{counter, histogram, Counter, Histogram};
use std::collections::HashSet;
use std::path::Path;
use tokio::time::Instant;
use tonic::{Request, Response, Status};

struct TimeHandle<'a> {
    histo: &'a Histogram,
    start: Instant,
}
impl<'a> TimeHandle<'a> {
    fn new(histo: &'a Histogram) -> TimeHandle<'a> {
        TimeHandle {
            start: Instant::now(),
            histo,
        }
    }
}

impl<'a> Drop for TimeHandle<'a> {
    fn drop(&mut self) {
        self.histo.record(self.start.elapsed().as_nanos() as f64);
    }
}

pub struct KnnController {
    countries: Vec<String>,
    knn_country: KnnByCountry,
    metrics: ControllerMetrics,
}
impl KnnController {
    pub fn new(countries: Vec<String>, config: KnnConfig) -> KnnController {
        let labels = [("service", "knn_controller")];
        let metrics = ControllerMetrics {
            request_count: counter!("request_count", &labels),
            request_latency: histogram!("request_latency", &labels),
        };
        KnnController {
            countries,
            knn_country: KnnByCountry::new(config),
            metrics,
        }
    }

    pub fn load(&mut self) -> anyhow::Result<()> {
        self.knn_country.load_countries(&self.countries)?;
        Ok(())
    }

    fn build_response(products: Vec<IndexResult>) -> KnnResponse {
        let mut response = KnnResponse::default();
        response.products = products
            .iter()
            .map(|ir| Product {
                product_id: ir.label,
                score: ir.distance,
                dotproduct: 0f32,
                squared_l2_norm: 0f32,
            })
            .collect();
        response
    }
}
struct ControllerMetrics {
    request_count: Counter,
    request_latency: Histogram,
}

#[tonic::async_trait]
impl Knn for KnnController {
    async fn search(&self, request: Request<KnnRequest>) -> Result<Response<KnnResponse>, Status> {
        self.metrics.request_count.increment(1);
        TimeHandle::new(&self.metrics.request_latency);
        let request: KnnRequest = request.into_inner();
        debug!("Received request with country: {}", request.country);
        if let Some(knn_service) = self.knn_country.get_service(&request.country) {
            let events: Vec<UserEvent> = request
                .user_events
                .iter()
                .map(|event| UserEvent {
                    index: event.partner_id,
                    label: event.product_id,
                    timestamp: event.timestamp as u64,
                    event_type: event.event_type,
                })
                .collect();
            let result = knn_service.get_closest_items(
                &events,
                request.index_id,
                request.result_count as usize,
                self.model.clone(),
            );

            match result {
                Ok(r) => {
                    let response = Response::new(KnnController::build_response(r));
                    Ok(response)
                }
                Err(error) => Err(Status::internal(error.to_string())),
            }
        } else {
            Err(Status::not_found(format!(
                "country {} not available",
                request.country
            )))
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
        let countries: Vec<CountryInfo> = self
            .knn_country
            .get_countries()
            .into_iter()
            .map(|c| {
                let mut country_info = CountryInfo::default();
                country_info.name = c;
                country_info
            })
            .collect();
        Ok(Response::new(AvailableCountriesResponse { countries }))
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
