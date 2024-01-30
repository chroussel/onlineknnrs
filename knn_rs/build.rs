fn main() {
    prost_build::compile_protos(&["proto/config.proto"], &["proto/"]).unwrap();
}
