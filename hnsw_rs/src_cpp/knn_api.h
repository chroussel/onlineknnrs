#pragma once
#include "hnswindex.h"

extern "C"
{
    typedef void *rust_hnsw_index_t;
    rust_hnsw_index_t create_index(Distance distance, const int dim);
    void init_new_index(rust_hnsw_index_t index, const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed);
    void save_index(rust_hnsw_index_t index, const char *path_to_index);
    void load_index(rust_hnsw_index_t index, const char *path_to_index);
    void set_ef(rust_hnsw_index_t index, const size_t ef);
    size_t cur_element_count(rust_hnsw_index_t index);
    bool get_data_pointer_by_label(rust_hnsw_index_t index, size_t label, float *dst);
    size_t query(rust_hnsw_index_t index, float *vector, size_t *items, float *distances, size_t k);
    void add_item(rust_hnsw_index_t index, float * vector, size_t id);
    void destroy(rust_hnsw_index_t index);
}
