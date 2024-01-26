#[macro_use]
extern crate criterion;
extern crate hnsw_rs;

use criterion::{Criterion, ParameterizedBenchmark};
use criterion::black_box;

use hnsw_rs::knnservice::*;
use hnsw_rs::hnswindex::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::str::FromStr;
use std::fs;
use hnsw_rs::{IndexConfig, Distance};
use hnsw_rs::embedding_computer::UserEvent;

const DIM: usize = 100;
const NB_EMBEDDINGS: usize = 50;
const INDEX_ID: i32 = 5;
const index_file_path:&str = "data/index-10k.hnsw";
const ids_file_path:&str = "data/input-ids-15k.csv";

fn bench(c: &mut Criterion) {
    let config = IndexConfig::new(Distance::Euclidean, DIM, 50);
    let index_file = PathBuf::from_str(index_file_path).expect("path");
    if !index_file.exists() {
        panic!("File {}", index_file_path)
    }
    let mut knn_service = KnnService::new(config);
    knn_service.add_index(INDEX_ID, index_file).expect("Loading index");
    let mut labels: Vec<UserEvent> = vec!();
    let data = fs::read_to_string(ids_file_path).expect("Unable to read file");
    for line in data.trim().split('\n') {
        for i in line.split(' ') {
            let v = i.parse().unwrap();
            labels.push(UserEvent {
                index: INDEX_ID,
                label: v,
                timestamp: 123,
                event_type: 2
            });
        }
    }
    c.bench_function("get_closest", move |b| {
        let mut cursor = 0;

        b.iter(||{
            cursor+=1;
            let data: Vec<UserEvent> = labels.iter().skip(cursor * NB_EMBEDDINGS).take(NB_EMBEDDINGS).map(|t|t.clone()).collect();
            let r = knn_service.get_closest_items(&data, INDEX_ID, 20, Model::Average);
            black_box(r)
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);