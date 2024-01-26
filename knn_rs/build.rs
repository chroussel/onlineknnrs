extern crate cc;
extern crate prost_build;

fn main() {
    cc::Build::new()
        .cpp(true)
        .include("includes")
        .file("src_cpp/knn_api.cpp")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-msse4")
        .flag_if_supported("-mavx")
        .flag_if_supported("/arch:AVX")
        .flag_if_supported("/arch:AVX2")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated")
        .compile("knnservice");

    prost_build::compile_protos(&["proto/config.proto"], &["proto/"]).unwrap();
}