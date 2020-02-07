#[macro_use]
extern crate criterion;
extern crate hnsw_rs;

use criterion::{Criterion, ParameterizedBenchmark};
use criterion::black_box;

use hnsw_rs::knnservice::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::str::FromStr;
use std::fs;

const DIM: i32 = 100;
const NB_EMBEDDINGS: usize = 50;
const INDEX_ID: i32 = 5;

fn bench(c: &mut Criterion) {
    let mut knn_service = KnnService::new(Distance::Euclidean, DIM, 50);
    let index_file =PathBuf::from_str("data/index-10k.hnsw").expect("path");
    knn_service.load_index(INDEX_ID, index_file).expect("Error loading index");
    let mut labels: Vec<(i32, i64)> = vec!();
    let data = fs::read_to_string("data/input-ids-15k.csv").expect("Unable to read file");
    for line in data.trim().split('\n') {
        for i in line.split(' ') {
            let v = i.parse().unwrap();
            labels.push((INDEX_ID, v));
        }
    }
    c.bench_function("get_closest", move |b| {
        let mut cursor = 0;

        b.iter(||{
            cursor+=1;
            let data = labels.iter().skip(cursor * NB_EMBEDDINGS).take(NB_EMBEDDINGS).map(|t|t.clone()).collect();
            let r = knn_service.get_closest_items(&data, INDEX_ID, 20, Model::Average);
            black_box(r)
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);