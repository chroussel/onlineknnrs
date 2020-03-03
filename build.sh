#!/usr/bin/env bash
set -e

cargo build
rm -rf "dist"
mkdir -p "dist"
cp "knn_py/examples/basic.py" "dist/."
find . -name "libtensorflow*.so" -exec cp {} dist/. \;
cp "target/debug/libknn_py.dylib" "dist/knn_py.so"

