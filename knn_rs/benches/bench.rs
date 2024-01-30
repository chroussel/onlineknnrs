#[macro_use]
extern crate criterion;
extern crate knn_rs;

use criterion::black_box;
use criterion::Criterion;

use knn_rs::embedding_computer::UserEvent;
use knn_rs::knncountry::{KnnByCountry, KnnConfig};
use knn_rs::knnservice::*;
use std::path::PathBuf;
use std::str::FromStr;

const NB_EMBEDDINGS: usize = 50;
const INDEX_ID: i32 = 868;

fn bench(c: &mut Criterion) {
    let config = KnnConfig {
        platform: "EU".into(),
        indices_root_path: PathBuf::from_str("data/all_indices").expect("path"),
        models: vec![Model {
            is_default: true,
            name: "avg".into(),
            model_type: ModelType::Average,
            version: None,
            model_path: None,
        }],
        version: "20240124000000".into(),
    };
    let mut kc = KnnByCountry::new(config);
    kc.load_countries(&["FR".into()]).expect("Loading index");

    let knn_service = kc.get_service("FR").expect("loaded country");

    let mut labels: Vec<UserEvent> = vec![];

    for v in knn_service.list_labels(5).expect("no issues") {
        labels.push(UserEvent {
            index: INDEX_ID,
            label: v,
            timestamp: 123,
            event_type: 2,
        });
    }

    c.bench_function("get_closest", move |b| {
        let mut cursor = 0;

        b.iter(|| {
            cursor += 1;
            let data: Vec<UserEvent> = labels
                .iter()
                .skip(cursor * NB_EMBEDDINGS)
                .take(NB_EMBEDDINGS)
                .map(|t| t.clone())
                .collect();
            let r = knn_service.get_closest_items(&data, INDEX_ID, 20, Some("avg".into()));
            black_box(r)
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
