use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{ListAccessor, Row, List};
use parquet::record::RowAccessor;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use crate::hnswindex::HnswIndex;
use crate::error::KnnError;
use std::fs;
use crate::Distance;
use std::convert::TryFrom;
use std::io::Write;

struct PartnerChunk {
    partner_id: i32,
    data_path: PathBuf,
}

pub enum Loader {}

impl Loader {
    pub fn load_extra_item_folder<P, F>(path: P, add_non_searchable_items: F) -> Result<(), KnnError>
        where
            P: AsRef<Path>,
            F: FnMut(i32, i64, Vec<f32>)
    {
        info!("Loading extra items from {}", path.display());
        let index_paths = fs::read_dir(path)?;

        for path in index_paths {
            let entry: fs::DirEntry = path?;
            if entry.file_name().into_string()?.starts_with("_") {
                continue;
            }
            info!("Loading path: {:?}", entry);
            Loader::parse_extra_items(&entry.path(), &add_non_searchable_items);
        }
        Ok(())
    }

    fn parse_extra_items<P, F>(path: P, add_non_searchable_items: F) -> Result<(), KnnError>
    where
        P: AsRef<Path>,
        F: FnMut(i32, i64, Vec<f32>) -> Result<(), KnnError>
    {
        let reader = SerializedFileReader::try_from(path.to_str()?)?;
        let mut iter = reader.get_row_iter(None)?;
        while let Some(record) = iter.next() {
            let record: Row = record;
            let product_partner: &Row = record.get_group(0)?;
            let partner_id = product_partner.get_int(1)?;
            let product = product_partner.get_long(0)?;
            let embedding_list: &List = record.get_list(1)?;
            let mut embedding = vec![];
            embedding.reserve_exact(embedding_list.len());
            for i in 0..embedding_list.len() {
                embedding.push(embedding_list.get_float(i)?);
            }
            add_non_searchable_items(partner_id, product, embedding)?;
        }
        Ok(())
    }

    pub fn load_index_folder<P>(path: P, add_index: F) -> Result<(), KnnError>
    where
        P: AsRef<Path>,
        F: FnMut(i32, PathBuf) ->  Result<(), KnnError>
    {
        let tempdir = TempDir::new("knn_index").unwrap();
        let path = path.as_ref();
        info!("Loading knn index from {}", path.display());
        let index_paths = fs::read_dir(path).expect("working path");
        let mut indexes_list = vec![];
        for path in index_paths {
            let entry: fs::DirEntry = path.expect("dir entry");
            if entry.file_name().into_string().unwrap().starts_with("_") {
                continue;
            }
            info!("Loading path: {:?}", entry);

            let partner_chunks = Loader::parse_index_file(entry.path(), tempdir.path());
            for pc in partner_chunks {
                add_index(pc.partner_id, pc.data_path)
            }
        }
        info!("Load done");
        Ok(())
    }

    fn parse_index_file<P: AsRef<Path>>(path: P, tempdir: P) -> Vec<PartnerChunk> {
        let path = path.as_ref();
        let tempdir = tempdir.as_ref();
        let reader = SerializedFileReader::try_from(path.to_str().unwrap()).unwrap();
        let mut pcs = vec![];
        let mut iter = reader.get_row_iter(None).unwrap();
        while let Some(record) = iter.next() {
            let partner_chunk = record.get_group(0).unwrap();
            let partner = partner_chunk.get_int(0).unwrap();
            let chunk = partner_chunk.get_int(1).unwrap();
            let ba = record.get_bytes(1).unwrap();
            let data_path = tempdir.join(format!("chunk-{}-{}", partner, chunk));
            let mut file = fs::File::create(&data_path).unwrap();
            file.write_all(ba.data()).unwrap();
            let pc = PartnerChunk {
                partner_id: partner,
                data_path,
            };
            pcs.push(pc);
        }
        pcs
    }
}