use crate::embedding_computer::{EmbeddingResult, UserEmbeddingComputer, UserEvent};
use crate::knnindex::EmbeddingRegistry;
use crate::KnnError;
use prost::Message;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::SystemTime;
use tensorflow::Tensor;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs};

pub struct KnnTf {
    graph: Graph,
    session: Session,
}

pub mod config {
    include!(concat!(env!("OUT_DIR"), "/tensorflow.rs"));
}

impl KnnTf {
    const PRODUCT_EMBEDDINGS: &'static str = "knn/feed/product_embeddings";
    const TIMESTAMPS: &'static str = "knn/feed/timestamps_sec";
    const CURRENT_TIMESTAMP: &'static str = "knn/feed/current_timestamp_sec";
    //const PUBLISHER_EMBEDDING:&'static str = "knn/feed/publisher_embedding";
    const NB_EVENT: &'static str = "knn/feed/nb_events";
    const EVENT_TYPES: &'static str = "knn/feed/event_types";
    const FETCH_NAME: &'static str = "knn/fetch/user_embedding";
    fn build_config() -> Result<SessionOptions, KnnError> {
        let config_proto = config::ConfigProto {
            experimental: Some(config::config_proto::Experimental {
                executor_type: "SINGLE_THREADED_EXECUTOR".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut buf = Vec::<u8>::with_capacity(config_proto.encoded_len());
        config_proto.encode_raw(&mut buf);
        let mut session = SessionOptions::new();
        session.set_config(&buf)?;
        Ok(session)
    }

    pub fn load_model<P: AsRef<Path>>(model_path: P) -> Result<KnnTf, KnnError> {
        let model_path = model_path.as_ref();
        let mut graph = Graph::new();
        let mut proto = Vec::new();
        File::open(model_path)?.read_to_end(&mut proto)?;
        let options = ImportGraphDefOptions::new();
        graph.import_graph_def(&proto, &options)?;
        let session_options = KnnTf::build_config()?;
        let session = Session::new(&session_options, &graph)?;
        Ok(KnnTf { graph, session })
    }
}

impl UserEmbeddingComputer for KnnTf {
    fn compute_user_vector(
        &self,
        registry: &EmbeddingRegistry,
        user_events: &[UserEvent],
    ) -> Result<EmbeddingResult, KnnError> {
        let mut product_embedding = Vec::with_capacity(registry.dim * user_events.len());
        let mut user_event_used = 0;
        for user_event in user_events {
            if let Some(mut emb) = registry.fetch_item(user_event.index, user_event.label)? {
                user_event_used += 1;
                product_embedding.append(&mut emb)
            } else {
                product_embedding.append(&mut vec![0f32; registry.dim])
            }
        }
        let product_tensor: Tensor<f32> =
            Tensor::new(&[1, user_events.len() as u64, registry.dim as u64])
                .with_values(&product_embedding)?;
        let event_count = Tensor::from(user_events.len() as i64);
        let current_timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let current_timestamp_tensor = Tensor::from(current_timestamp as i64);
        let timestamps: Vec<i64> = user_events.iter().map(|u| u.timestamp as i64).collect();
        let timestamps_tensor =
            Tensor::new(&[1, user_events.len() as u64]).with_values(timestamps.as_slice())?;
        let event_types: Vec<i64> = user_events.iter().map(|u| u.event_type as i64).collect();
        let event_type_tensor =
            Tensor::new(&[1, user_events.len() as u64]).with_values(event_types.as_slice())?;

        let mut session_args = SessionRunArgs::new();
        if let Some(op) = self
            .graph
            .operation_by_name(KnnTf::PRODUCT_EMBEDDINGS)
            .unwrap()
        {
            session_args.add_feed(&op, 0, &product_tensor);
        }
        if let Some(op) = self
            .graph
            .operation_by_name(KnnTf::CURRENT_TIMESTAMP)
            .unwrap()
        {
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

        let fetch = session_args.request_fetch(
            &self.graph.operation_by_name_required(KnnTf::FETCH_NAME)?,
            0,
        );
        self.session.run(&mut session_args)?;
        let result_tensor: Tensor<f32> = session_args.fetch(fetch)?;
        let result_emb = result_tensor.to_vec();

        Ok(EmbeddingResult {
            user_embedding: result_emb,
            user_event_used_count: user_event_used,
        })
    }
}
