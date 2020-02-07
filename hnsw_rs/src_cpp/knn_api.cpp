#include "hnswindex.h"
#include "knn_api.h"

typedef Index<float> *index_ptr;

rust_hnsw_index_t create_index(Distance distance, const int dim)
{
    return new Index<float>(distance, dim);
}

index_ptr cast(rust_hnsw_index_t index)
{
    return static_cast<index_ptr>(index);
}

void init_new_index(rust_hnsw_index_t index, const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed)
{
    cast(index)->initNewIndex(maxElements, M, efConstruction, random_seed);
}

void save_index(rust_hnsw_index_t index, const char *path_to_index)
{
    cast(index)->saveIndex(path_to_index);
}

void load_index(rust_hnsw_index_t index, const char *path_to_index)
{
    cast(index)->loadIndex(path_to_index);
}

void set_ef(rust_hnsw_index_t index, const size_t ef)
{
    cast(index)->appr_alg->setEf(ef);
}

size_t cur_element_count(rust_hnsw_index_t index)
{
    return cast(index)->appr_alg->cur_element_count;
}

bool get_data_pointer_by_label(rust_hnsw_index_t index, size_t label, float *dst)
{
    return cast(index)->getDataPointerByLabel(label, dst);
}

size_t query(rust_hnsw_index_t index, float *vector, size_t *items, float *distances, size_t k)
{
    return cast(index)->knnQuery(vector, items, distances, k);
}

void add_item(rust_hnsw_index_t index, float * vector, size_t id) {
    return cast(index)->addItem(vector, id);
}

void destroy(rust_hnsw_index_t index) {
    delete cast(index);
}