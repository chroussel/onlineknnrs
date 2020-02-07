pub const Distance_Euclidian: Distance = 1;
pub const Distance_Angular: Distance = 2;
pub const Distance_InnerProduct: Distance = 3;

pub type Distance = i32;
pub type rust_hnsw_index_t = u64;

extern "C" {
    pub fn create_index(distance: Distance, dim: ::std::os::raw::c_int) -> rust_hnsw_index_t;

    pub fn init_new_index(
        index: rust_hnsw_index_t,
        max_elements: usize,
        M: usize,
        efConstruction: usize,
        random_seed: usize,
    );


    pub fn save_index(index: rust_hnsw_index_t, path_to_index: *const ::std::os::raw::c_char);


    pub fn load_index(index: rust_hnsw_index_t, path_to_index: *const ::std::os::raw::c_char);


    pub fn set_ef(index: rust_hnsw_index_t, ef: usize);


    pub fn cur_element_count(index: rust_hnsw_index_t) -> usize;


    pub fn get_data_pointer_by_label(index: rust_hnsw_index_t, label: usize, dst: *mut f32)
                                     -> bool;

    pub fn query(
        index: rust_hnsw_index_t,
        vector: *mut f32,
        items: *mut usize,
        distances: *mut f32,
        k: usize,
    ) -> usize;

    pub fn add_item(index: rust_hnsw_index_t, vector: *mut f32, label: usize);
    pub fn destroy(index: rust_hnsw_index_t);
}