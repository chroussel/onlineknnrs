SHELL := /bin/bash

ts := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
.PHONY: help
help: ## This help message
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

.PHONY: build
build: nightly ## Builds Rust code and knn_py Python modules
	cargo build

.PHONY: build-release
build-release: nightly ## Build knn_py module in release mode
	cargo build --release

.PHONY: python-build
python-build: nightly dev-packages
	(cd knn_python && python setup.py build)

.PHONY: nightly
nightly: ## Set rust compiler to nightly version
	rustup override set nightly

.PHONY: install
install: nightly dev-packages ## Install knn_py module into current python enviroment
	(cd knn_python && python setup.py install)

.PHONY: clean-python
clean-python: ## Clean up python build artifacts
	(cd knn_python && rm -rf build dist *.egg-info external)

.PHONY: clean
clean: clean-python ## Clean up build artifacts
	cargo clean

.PHONY: dev-packages
dev-packages: ## Install Python development packages for project
	pip install -r knn_python/requirements.txt

.PHONY: test
test: ## Run tests
	cargo test