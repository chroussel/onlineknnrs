use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{ListAccessor, Row, List};
use parquet::record::RowAccessor;
use std::path::{Path, PathBuf};
use failure::Error;
use std::fs;
use std::convert::TryFrom;
use std::io::Write;
use ndarray::Array1;
use tempdir::TempDir;

struct PartnerChunk {
    partner_id: i32,
    data_path: PathBuf,
}

pub enum Loader {}

impl Loader {
    pub fn load_extra_item_folder<P, F>(path: P, mut add_non_searchable_items: F) -> Result<(), Error>
        where
            P: AsRef<Path>,
            F: FnMut(i32, i64, Array1<f32>)
    {
        info!("Loading extra items from {}", path.as_ref().display());
        let index_paths = fs::read_dir(path)?;

        for path in index_paths {
            let entry: fs::DirEntry = path?;
            if entry.file_name().into_string().unwrap().starts_with('_') {
                continue;
            }
            info!("Loading path: {:?}", entry);
            Loader::parse_extra_items(&entry.path(), &mut add_non_searchable_items)?;
        }
        Ok(())
    }

    fn parse_extra_items<P, F>(path: P, mut add_non_searchable_items: F) -> Result<(), Error>
    where
        P: AsRef<Path>,
        F: FnMut(i32, i64, Array1<f32>)
    {
        let reader = SerializedFileReader::try_from(path.as_ref())?;
        for record in reader.get_row_iter(None)? {
            let product_partner: &Row = record.get_group(0)?;
            let partner_id = product_partner.get_int(1)?;
            let product = product_partner.get_long(0)?;
            let embedding_list: &List = record.get_list(1)?;
            let mut embedding = Array1::<f32>::zeros(embedding_list.len() + 1);
            for i in 0..embedding_list.len() {
                embedding[i] = embedding_list.get_float(i)?;
            }
            add_non_searchable_items(partner_id, product, embedding);
        }
        Ok(())
    }

    pub fn load_index_folder<P, F>(path: P, mut add_index: F) -> Result<(), Error>
    where
        P: AsRef<Path>,
        F: FnMut(i32, PathBuf) ->  Result<(), Error>
    {
        let tempdir = TempDir::new("knn_index").unwrap();
        let path = path.as_ref();
        info!("Loading knn index from {}", path.display());
        let index_paths = fs::read_dir(path).expect("working path");
        for path in index_paths {
            let entry: fs::DirEntry = path.expect("dir entry");
            if entry.file_name().into_string().unwrap().starts_with('_') {
                continue;
            }
            info!("Loading path: {}", entry.path().display());

            let partner_chunks = Loader::parse_index_file(entry.path(), tempdir.path())?;
            for pc in partner_chunks {
                add_index(pc.partner_id, pc.data_path)?;
            }
        }
        info!("Load done");
        Ok(())
    }

    fn parse_index_file<P1: AsRef<Path>, P2: AsRef<Path>>(path: P1, tempdir: P2) -> Result<Vec<PartnerChunk>, Error> {
        let path = path.as_ref();
        let tempdir = tempdir.as_ref();
        let reader = SerializedFileReader::try_from(path.to_str().unwrap())?;
        let mut pcs = vec![];
        for record in reader.get_row_iter(None)? {
            let partner_chunk = record.get_group(0)?;
            let partner = partner_chunk.get_int(0)?;
            let chunk = partner_chunk.get_int(1)?;
            let ba = record.get_bytes(1)?;
            let data_path = tempdir.join(format!("chunk-{}-{}", partner, chunk));
            let mut file = fs::File::create(&data_path)?;
            file.write_all(ba.data())?;
            let pc = PartnerChunk {
                partner_id: partner,
                data_path,
            };
            pcs.push(pc);
        }
        Ok(pcs)
    }
}