#!/usr/bin/env bash
set -e

OS="$1"
VERSION="$2"
DESTINATION="$3"

TF_URL="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-${OS}-x86_64-${VERSION}.tar.gz"
echo "Downloading ${TF_URL} to $DESTINATION"
mkdir -p "$DESTINATION"
(
  cd $DESTINATION
  curl -o tensorflow.tar.gz "$TF_URL"
  tar xzf tensorflow.tar.gz
  rm tensorflow.tar.gz
)
