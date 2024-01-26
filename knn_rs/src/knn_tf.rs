use std::path::Path;
use failure::Error;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs};
use std::fs::File;
use std::io::Read;
use ndarray::{Axis, Array2, ArrayView1};
use tensorflow::Tensor;
use crate::embedding_computer::{UserEmbeddingComputer, EmbeddingResult, UserEvent};
use crate::knnindex::EmbeddingRegistry;
use crate::KnnError;
use std::ops::Deref;
use std::time::SystemTime;
use prost::Message;

pub struct KnnTf {
    graph: Graph,
    session: Session
}

pub mod config {
    include!(concat!(env!("OUT_DIR"), "/tensorflow.rs"));
}



impl KnnTf {
    const PRODUCT_EMBEDDINGS:&'static str = "knn/feed/product_embeddings";
    const TIMESTAMPS:&'static str = "knn/feed/timestamps_sec";
    const CURRENT_TIMESTAMP:&'static str = "knn/feed/current_timestamp_sec";
    //const PUBLISHER_EMBEDDING:&'static str = "knn/feed/publisher_embedding";
    const NB_EVENT:&'static str = "knn/feed/nb_events";
    const EVENT_TYPES:&'static str = "knn/feed/event_types";
    const FETCH_NAME: &'static str  = "knn/fetch/user_embedding";
    fn build_config() -> Result<SessionOptions, KnnError> {
        let mut exp = config::config_proto::Experimental::default();
        exp.executor_type = String::from("SINGLE_THREADED_EXECUTOR");
        let mut config_proto = config::ConfigProto::default();
        config_proto.experimental = Some(exp);
        let mut buf = Vec::<u8>::with_capacity(config_proto.encoded_len());
        config_proto.encode_raw(&mut buf);
        let mut session = SessionOptions::new();
        session.set_config(&buf)
            .map_err(KnnError::from)?;
        Ok(session)
    }

    pub fn load_model<P: AsRef<Path>>(model_path: P) -> Result<KnnTf, Error> {
        let model_path = model_path.as_ref();
        let mut graph = Graph::new();
        let mut proto = Vec::new();
        File::open(model_path)?.read_to_end(&mut proto)?;
        let options = ImportGraphDefOptions::new();
        graph.import_graph_def(&proto, &options)
            .map_err(KnnError::from)?;
        let session_options = KnnTf::build_config()
            .map_err(KnnError::from)?;
        let session = Session::new(&session_options, &graph)
            .map_err(KnnError::from)?;
        Ok(KnnTf {
            graph,
            session
        })
    }
}

impl UserEmbeddingComputer for KnnTf {
    fn compute_user_vector(&self, registry: &EmbeddingRegistry, user_events: &[UserEvent]) -> Result<EmbeddingResult, KnnError> {
        let mut product_embedding = Array2::<f32>::zeros((registry.dim, user_events.len()));
        let mut user_event_used = 0;
        user_events.iter()
            .zip(product_embedding.lanes_mut(Axis(0)))
            .for_each(|(user_event, mut row)| {
                let emb = registry.fetch_item(user_event.index, user_event.label)
                    .and_then(|e| { user_event_used+=1; Some(e)})
                    .unwrap_or_else(|| registry.zero());
                row.assign(&emb)
        });
        let product_tensor: Tensor<f32> = Tensor::new(&[1, user_events.len() as u64, registry.dim as u64])
            .with_values(product_embedding.as_slice().unwrap())?;
        let event_count = Tensor::from(user_events.len() as i64);
        let current_timestamp = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
        let current_timestamp_tensor = Tensor::from(current_timestamp as i64);
        let timestamps: Vec<i64> = user_events.iter().map(|u|u.timestamp as i64).collect();
        let timestamps_tensor = Tensor::new(&[1, user_events.len() as u64])
            .with_values(timestamps.as_slice())?;
        let event_types: Vec<i64> = user_events.iter().map(|u|u.event_type as i64).collect();
        let event_type_tensor = Tensor::new(&[1, user_events.len() as u64])
            .with_values(event_types.as_slice())?;

        let mut session_args = SessionRunArgs::new();
        if let Some(op) = self.graph.operation_by_name(KnnTf::PRODUCT_EMBEDDINGS).unwrap() {
            session_args.add_feed(&op, 0, &product_tensor);
        }
        if let Some(op) = self.graph.operation_by_name(KnnTf::CURRENT_TIMESTAMP).unwrap() {
            session_args.add_feed(&op, 0, &current_timestamp_tensor);
        }
        if let Some(op) = self.graph.operation_by_name(KnnTf::TIMESTAMPS).unwrap() {
            session_args.add_feed(&op, 0, &timestamps_tensor);
        }
        if let Some(op) = self.graph.operation_by_name(KnnTf::EVENT_TYPES).unwrap() {
            session_args.add_feed(&op, 0, &event_type_tensor);
        }
        if let Some(op) = self.graph.operation_by_name(KnnTf::NB_EVENT).unwrap() {
            session_args.add_feed(&op, 0, &event_count);
        }

        let fetch = session_args.request_fetch(&self.graph.operation_by_name_required(KnnTf::FETCH_NAME)?, 0);
        self.session.run(&mut session_args)?;
        let result_tensor: Tensor<f32> = session_args.fetch(fetch)?;
        let slice:&[f32] = result_tensor.deref();
        let result_emb = ArrayView1::<f32>::from(slice);
        Ok(EmbeddingResult {
            user_embedding: result_emb.into_owned(),
            user_event_used_count: user_event_used
        })
    }
}
