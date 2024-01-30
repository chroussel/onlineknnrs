use byteorder::{BigEndian, ReadBytesExt};
use faiss::Idx;
use std::collections::HashMap;
use std::io::{BufReader, ErrorKind};
use std::path::Path;

use crate::knnindex::{KnnIndex, Metadata};
use crate::wrappedindex::WrappedIndex;
use crate::KnnError;

pub enum Loader {}

impl Loader {
    fn load_embedding_norms<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, KnnError> {
        let f = std::fs::File::open(path)?;
        let len = f.metadata().map(|m| m.len()).unwrap_or(0u64);
        let mut vec = Vec::with_capacity((len / 4) as usize);
        let mut buf = BufReader::new(f);
        loop {
            match buf.read_f32::<BigEndian>() {
                Ok(v) => {
                    vec.push(v);
                }
                Err(e) => {
                    if e.kind() == ErrorKind::UnexpectedEof {
                        // Nothing more to read
                        break;
                    } else {
                        return Err(KnnError::IoError(e));
                    }
                }
            }
        }
        Ok(vec)
    }

    fn load_mapping<P: AsRef<Path>>(path: P) -> Result<HashMap<i64, faiss::Idx>, KnnError> {
        let f = std::fs::File::open(path)?;
        let len = f.metadata().map(|m| m.len()).unwrap_or(0u64);
        let mut map = HashMap::with_capacity((len / 8) as usize);
        let mut buf = BufReader::new(f);
        let mut idx = 0;
        loop {
            match buf.read_i64::<BigEndian>() {
                Ok(v) => {
                    map.insert(v, Idx::new(idx));
                    idx += 1;
                }
                Err(e) => {
                    if e.kind() == ErrorKind::UnexpectedEof {
                        // Nothing more to read
                        break;
                    } else {
                        return Err(KnnError::IoError(e));
                    }
                }
            }
        }
        Ok(map)
    }

    fn load_index<P: AsRef<Path>>(path: P, metadata: &Metadata) -> Result<WrappedIndex, KnnError> {
        let path = path.as_ref();
        let f_boolean = if metadata.is_recommendable {
            "True"
        } else {
            "False"
        };
        let index_filename = format!(
            "{}.{}.{}.{}.index",
            metadata.country, metadata.partner_id, metadata.chunk_id, f_boolean
        );
        let norm_filename = format!("{}_embeddingNorms.array", index_filename);
        let mapping_filename = format!("{}_inverseMapping.array", index_filename);
        let local_path = path.join("indices").join(index_filename);

        let local_path_str = local_path
            .into_os_string()
            .into_string()
            .map_err(|_| KnnError::InvalidPath)?;
        let index = faiss::read_index(local_path_str)?;

        let mapping = Loader::load_mapping(path.join("indices").join(mapping_filename.clone()))?;
        let norm = Loader::load_embedding_norms(path.join("indices").join(mapping_filename))?;
        Ok(WrappedIndex::new(Box::new(index), mapping, norm))
    }

    pub fn load_index_folder<P>(path: P) -> Result<HashMap<i32, KnnIndex>, KnnError>
    where
        P: AsRef<Path>,
    {
        info!("Starting to load {}", path.as_ref().display());
        let metadata_path = path.as_ref().join("metadata.json");
        let fs = std::fs::File::open(metadata_path)?;
        let metadatas: Vec<Metadata> = serde_json::from_reader(fs)?;
        let mut indices: HashMap<i32, KnnIndex> = HashMap::new();

        for m in metadatas {
            debug!("Loading chunk {}/{}", m.partner_id, m.chunk_id);
            let index = Loader::load_index(path.as_ref(), &m)?;
            let mut ki = indices
                .entry(m.partner_id)
                .or_insert_with(|| KnnIndex::new());
            if m.is_recommendable {
                ki.add_reco_index(index)
            } else {
                ki.add_non_reco_index(index)
            }
        }
        info!("Load done");
        Ok(indices)
    }
}
