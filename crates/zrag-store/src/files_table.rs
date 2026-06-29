use anyhow::Result;
use arrow::array::{FixedSizeBinaryArray, RecordBatch, StringArray, UInt64Array};
use lancedb::query::ExecutableQuery;
use lancedb::table::Table;
use std::sync::Arc;

use crate::schema;

pub struct FilesTable {
    table: Table,
}

impl FilesTable {
    pub async fn open(db: &lancedb::Connection) -> Result<Self> {
        let name = "files";
        let table = if db
            .table_names()
            .execute()
            .await?
            .contains(&name.to_string())
        {
            db.open_table(name).execute().await?
        } else {
            let schema = Arc::new(schema::files_schema());
            db.create_empty_table(name, schema).execute().await?
        };
        Ok(Self { table })
    }

    pub async fn upsert(&self, batch: RecordBatch) -> Result<()> {
        crate::upsert::upsert_batch(&self.table, "file_path", batch).await
    }

    pub async fn list(&self) -> Result<Vec<FileRow>> {
        let results = self.table.query().execute().await?;
        let total = self.table.count_rows(None).await?;
        let mut rows = Vec::with_capacity(total);
        let mut stream = std::pin::pin!(results);
        use futures::StreamExt;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let num_rows = batch.num_rows();

            let file_paths = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let blake3s = batch
                .column_by_name("blake3")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>());
            let mtimes = batch
                .column_by_name("mtime_ns")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
            let sizes = batch
                .column_by_name("size_bytes")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
            let languages = batch
                .column_by_name("language")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let indexed_ats = batch
                .column_by_name("indexed_at_ns")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());

            for i in 0..num_rows {
                rows.push(FileRow {
                    file_path: file_paths
                        .map(|a| a.value(i).to_string())
                        .unwrap_or_default(),
                    blake3: blake3s.map(|a| a.value(i).to_vec()).unwrap_or_default(),
                    mtime_ns: mtimes.map(|a| a.value(i)).unwrap_or(0),
                    size_bytes: sizes.map(|a| a.value(i)).unwrap_or(0),
                    language: languages
                        .map(|a| a.value(i).to_string())
                        .unwrap_or_default(),
                    chunk_ids: Vec::new(),
                    indexed_at_ns: indexed_ats.map(|a| a.value(i)).unwrap_or(0),
                });
            }
        }
        Ok(rows)
    }

    pub async fn delete_for_paths(&self, paths: &[&str]) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        for filter in crate::delete_filter::file_path_delete_filters(paths) {
            self.table.delete(&filter).await?;
        }
        Ok(())
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FileRow {
    pub file_path: String,
    pub blake3: Vec<u8>,
    pub mtime_ns: u64,
    pub size_bytes: u64,
    pub language: String,
    pub chunk_ids: Vec<Vec<u8>>,
    pub indexed_at_ns: u64,
}
