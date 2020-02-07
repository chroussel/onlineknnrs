extern crate cc;

use std::env;
use std::path::PathBuf;

fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src_cpp/knn_api.cpp")
        .flag_if_supported("-mavx")
        .flag_if_supported("/arch:avx")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated")
        .compile("knnservice");

    /*let bindings = bindgen::Builder::default()
        .rustfmt_bindings(true)
        .header("src_cpp/knn_api.h")
        .opaque_type("rust_hnsw_index_t")
        .clang_arg(r"-xc++")
        .clang_arg(r"-lsrdc++")
        .whitelist_recursively(false)
        .whitelist_type("rust_hnsw_index_t")
        .whitelist_type("Distance")
        .whitelist_function("create_index")
        .whitelist_function("init_new_index")
        .whitelist_function("save_index")
        .whitelist_function("load_index")
        .whitelist_function("set_ef")
        .whitelist_function("cur_element_count")
        .whitelist_function("get_data_pointer_by_label")
        .whitelist_function("query")
        .whitelist_function("destroy")
        .derive_copy(false)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");*/
}