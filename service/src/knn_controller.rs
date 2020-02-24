use std::collections::HashSet;
use hnsw_rs::knncountry::KnnByCountry;
use failure::Error;
use std::path::Path;
use tonic::{Response, Request, Status};
use hnsw_rs::knnservice::Model;
use crate::knn::{*, knn_server::*};
use dipstick::{AtomicBucket, InputScope, Observe, CancelHandle};
use hdrhistogram::Histogram;
use std::time::{Duration, Instant};
use std::sync::{Arc, RwLock};

struct LatencyHistogram {
    period: Duration,
    histograms: Arc<RwLock<Histogram<u64>>>
}

struct TimeHandle<'a> {
    now: std::time::Instant,
    histo: &'a LatencyHistogram
}

impl<'a> TimeHandle<'a> {
    fn start(histo: &LatencyHistogram) -> TimeHandle {
        TimeHandle{
            now: Instant::now(),
            histo
        }
    }

    fn stop(self) {
        let elapsed = self.now.elapsed().as_nanos();
        self.histo.histograms.write().unwrap().record(elapsed as u64).unwrap();
    }
}

impl LatencyHistogram {
    fn new(period: Duration) -> LatencyHistogram{
        LatencyHistogram {
            period,
            histograms: Arc::new(RwLock::new(Histogram::<u64>::new_with_max(1_000_000_000, 4).unwrap()))
        }
    }

    fn observe(&self, metrics: AtomicBucket, name: &str, quantile: f64) {
        let histo = self.histograms.clone();
        metrics
            .observe(metrics.gauge(&name), move |_now| histo.read().unwrap().value_at_quantile(quantile) as isize)
            .on_flush();
    }

    fn observe_mean_max(&self, metrics: AtomicBucket) {
        let histo = self.histograms.clone();
        metrics
            .observe(metrics.gauge("request.latency.mean"), move |_now| histo.read().unwrap().mean() as isize)
            .on_flush();
        let histo2 = self.histograms.clone();
        metrics
            .observe(metrics.gauge("request.latency.max"), move |_now| histo2.read().unwrap().max() as isize)
            .on_flush();
    }

    fn record(&self) -> TimeHandle {
        TimeHandle::start(self)
    }
}


pub struct KnnController {
    countries: HashSet<String>,
    knn_country:KnnByCountry,
    metrics: AtomicBucket,
    latency_histo: LatencyHistogram
}
impl KnnController {
    pub fn new(countries: Vec<String>, metrics: AtomicBucket) -> KnnController {
        let latency_histo = LatencyHistogram::new(Duration::from_secs(1));
        latency_histo.observe(metrics.clone(),"request.latency.50", 0.50);
        latency_histo.observe(metrics.clone(),"request.latency.90", 0.90);
        latency_histo.observe(metrics.clone(),"request.latency.99", 0.99);
        latency_histo.observe(metrics.clone(),"request.latency.999", 0.999);
        latency_histo.observe_mean_max(metrics.clone());
        KnnController {
            metrics,
            latency_histo,
            countries: countries.into_iter().map(|c| c.to_uppercase()).collect(),
            knn_country: KnnByCountry::default(),
        }
    }

    pub fn load<P>(&mut self, index_path: P, extra_item_path: P) -> Result<(), Error>
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
            Product {
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
        self.metrics.marker("request.count").mark();
        let handle = self.latency_histo.record();
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
                    let response = Response::new(KnnController::build_response(r));
                    handle.stop();
                    Ok(response)
                },
                Err(error) => {
                    handle.stop();
                    Err(Status::internal(error.to_string()))
                }
            }
        } else {
            handle.stop();
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