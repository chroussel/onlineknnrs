use tonic::{Request, Status, Response};
use crate::health_check::health::health_check_response::ServingStatus;
use crate::health_check::health::*;
use std::collections::HashMap;
use std::sync::{RwLock, Arc};

pub mod health {
    tonic::include_proto!("grpc.health.v1");
}

pub struct HealthCheckController {
    status_map: Arc<RwLock<HashMap<String, ServingStatus>>>
}

impl HealthCheckController {
    fn new() -> Self {
        HealthCheckController {
            status_map: Arc::new(RwLock::new(HashMap::new()))
        }
    }

    fn get_status(&self, service_name: &str) -> Option<ServingStatus> {
        self.status_map.read().ok()
            .map(|h| h.get(service_name))
    }

    fn set_status(&self, service_name: &str, status: ServingStatus) {
        if let Some(mut hash_map) = self.status_map.write().ok() {
            let entry = hash_map.entry(service_name.to_string())
                .or_insert(ServingStatus::Unknown);
            *entry = status;
        }
    }

    fn shutdown(&mut self) {
        if let Some(mut hash_map) = self.status_map.write().ok() {
            hash_map.iter_mut().for_each(|(key, value)| {
                *value = ServingStatus::NotServing
            })
        }
    }
}


#[tonic::async_trait]
impl health_server::Health for HealthCheckController {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let request: HealthCheckRequest = request.into_inner();
        let service_name = request.service;
        let status = self.get_status(service_name).unwrap_or(ServingStatus::Unknown);
        let response = HealthCheckResponse::default();
        response.set_status(status);
        Ok(response)
    }

    async fn watch(&self, request: Request<HealthCheckRequest>) -> Result<Response<Self::WatchStream>, Status> {
        Err(Status::unimplemented("watch"))
    }
}