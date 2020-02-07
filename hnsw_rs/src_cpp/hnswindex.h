#pragma once

#include <iostream>
#include <tuple>
#include <string>
#include "hnswlib.h"

enum Distance {
    Euclidian = 1,
    Angular = 2,
    InnerProduct = 3
};

template<typename dist_t, typename data_t=float>
class Index {
public:
    Index(hnswlib::SpaceInterface<float> *space, const int dim, bool normalize = false) :
        space(space), dim(dim), normalize(normalize) {
        appr_alg = NULL;
    }

    Index(Distance distance, const int dim) : dim(dim) {
        if(distance == Euclidian) {
            space = new hnswlib::L2Space(dim);
        }
        else if (distance == Angular) {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        }
        else if (distance == InnerProduct) {
            space = new hnswlib::InnerProductSpace(dim);
        }
        else {
            throw std::runtime_error("Distance not supported: " + std::to_string(distance));
        }
        appr_alg = NULL;
    }

    void initNewIndex(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, maxElements, M, efConstruction, random_seed);
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index) {
        if (appr_alg) {
            std::cerr<<"Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(space, path_to_index, false, 0);
    }

    void normalizeVector(dist_t *data, dist_t *norm_array){
        dist_t norm=0.0f;
        for(int i=0;i<dim;i++)
            norm+=data[i]*data[i];
        norm= 1.0f / (sqrtf(norm) + 1e-30f);
        for(int i=0;i<dim;i++)
            norm_array[i]=data[i]*norm;
    }

    void addItem(dist_t * vector, size_t id) {
        dist_t* vector_data = vector;
        std::vector<dist_t> norm_array(dim);
        if(normalize) {
            normalizeVector(vector_data, norm_array.data());
            vector_data = norm_array.data();
        }
        appr_alg->addPoint(vector_data, (size_t) id);
    }

    bool getDataPointerByLabel(size_t label, data_t* dst) {
        hnswlib::tableint label_c;
        auto search = appr_alg->label_lookup_.find(label);
        if (search == appr_alg->label_lookup_.end()) {
            return false;
        }
        label_c = search->second;
        data_t* data_ptr = (data_t*)appr_alg->getDataByInternalId(label_c);
        memcpy(dst, data_ptr, appr_alg->data_size_);
        return true;
    }

    std::vector<size_t> getIdsList() {
        std::vector<size_t> ids;

        for(auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    size_t knnQuery(dist_t * vector, size_t * items, dist_t * distances, size_t k) {
        dist_t* vector_data = vector;
        std::vector<dist_t> norm_array(dim);
        if(normalize) {
            normalizeVector(vector_data, norm_array.data());
            vector_data = norm_array.data();
        }

        std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                (void *) vector_data, k);
        size_t nbResults = result.size();
        if (nbResults != k)
            std::cout << "Retrieved " << nbResults << " items instead of " <<
                k << ". Items in the index: " << appr_alg->cur_element_count;

        for (int i = nbResults - 1; i >= 0; i--) {
            auto &result_tuple = result.top();
            distances[i] = result_tuple.first;
            items[i] = (size_t)result_tuple.second;
            result.pop();
        }
        return nbResults;
    }

    hnswlib::SpaceInterface<float> *space;
    int dim;
    bool normalize;
    hnswlib::HierarchicalNSW<dist_t> *appr_alg;

    ~Index() {
        delete space;
        if (appr_alg)
            delete appr_alg;
    }
};