pub const Distance_Euclidian: Distance = 1;
pub const Distance_Angular: Distance = 2;
pub const Distance_InnerProduct: Distance = 3;

pub type Distance = i32;
pub type RustHnswIndexT = u64;

extern "C" {
    pub fn create_index(distance: Distance, dim: ::std::os::raw::c_int) -> RustHnswIndexT;

    pub fn init_new_index(
        index: RustHnswIndexT,
        max_elements: usize,
        M: usize,
        efConstruction: usize,
        random_seed: usize,
    );


    pub fn save_index(index: RustHnswIndexT, path_to_index: *const ::std::os::raw::c_char);


    pub fn load_index(index: RustHnswIndexT, path_to_index: *const ::std::os::raw::c_char);


    pub fn set_ef(index: RustHnswIndexT, ef: usize);


    pub fn cur_element_count(index: RustHnswIndexT) -> usize;


    pub fn get_item(index: RustHnswIndexT, label: usize) -> *mut f32;

    pub fn query(
        index: RustHnswIndexT,
        vector: *mut f32,
        items: *mut usize,
        distances: *mut f32,
        k: usize,
    ) -> usize;

    pub fn add_item(index: RustHnswIndexT, vector: *mut f32, label: usize);
    pub fn destroy(index: RustHnswIndexT);
}