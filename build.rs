fn main() {
    tonic_build::compile_protos("proto/knn.proto").unwrap();
}