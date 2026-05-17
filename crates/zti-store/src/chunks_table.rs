use anyhow::Result;
use arrow::array::{BinaryArray, FixedSizeBinaryArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use lancedb::index::Index;
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use std::sync::Arc;

use crate::schema;

pub struct ChunksTable {
    table: Table,
    dim: usize,
    index_created: bool,
}

impl ChunksTable {
    pub async fn open(db: &lancedb::Connection, dim: usize) -> Result<Self> {
        let name = "chunks";
        let table = if db.table_names().execute().await?.contains(&name.to_string()) {
            db.open_table(name).execute().await?
        } else {
            let schema = Arc::new(schema::chunks_schema(dim));
            db.create_empty_table(name, schema).execute().await?
        };
        Ok(Self {
            table,
            dim,
            index_created: false,
        })
    }

    pub async fn upsert(&self, batch: RecordBatch) -> Result<()> {
        let schema = batch.schema();
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

        let mut builder = self.table.merge_insert(&["chunk_id"]);
        builder.when_matched_update_all(None);
        builder.when_not_matched_insert_all();
        builder.execute(reader).await?;

        Ok(())
    }

    pub async fn index_vector(&mut self) -> Result<()> {
        if self.index_created {
            return Ok(());
        }
        if self.table.count_rows(None).await? == 0 {
            return Ok(());
        }
        self.table
            .create_index(&["embedding"], Index::IvfPq(IvfPqIndexBuilder::default()))
            .execute()
            .await?;
        self.index_created = true;
        Ok(())
    }

    pub async fn delete_for_files(&self, paths: &[&str]) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        let filter = paths
            .iter()
            .map(|p| format!("file_path = '{}'", p))
            .collect::<Vec<_>>()
            .join(" OR ");
        self.table.delete(&filter).await?;
        Ok(())
    }

    pub async fn knn(
        &self,
        query: &[f32],
        k: usize,
        languages: Option<&[String]>,
        path_glob: Option<&str>,
    ) -> Result<Vec<ChunkHit>> {
        let mut q = self
            .table
            .query()
            .nearest_to(query)?
            .limit(k);

        let mut filters = Vec::new();

        if let Some(langs) = languages {
            if !langs.is_empty() {
                let list = langs
                    .iter()
                    .map(|l| format!("'{}'", l))
                    .collect::<Vec<_>>()
                    .join(",");
                filters.push(format!("language IN ({})", list));
            }
        }

        if let Some(glob) = path_glob {
            if let Some(pattern) = glob_to_like(glob) {
                filters.push(format!("file_path LIKE '{}'", pattern));
            }
        }

        if !filters.is_empty() {
            let combined = filters.join(" AND ");
            q = q.only_if(combined);
        }

        let results = q.execute().await?;

        let mut hits = Vec::new();
        let mut stream = std::pin::pin!(results);
        use futures::StreamExt;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let num_rows = batch.num_rows();
            let file_paths = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let start_lines = batch
                .column_by_name("start_line")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let end_lines = batch
                .column_by_name("end_line")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let contents = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let symbol_qualified = batch
                .column_by_name("symbol_qualified")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let chunk_ids = batch
                .column_by_name("chunk_id")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>());
            let turbo_codes = batch
                .column_by_name("turbo_code")
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..num_rows {
                let d = distances.map(|a| a.value(i)).unwrap_or(0.0);
                let score = 1.0 - d / 2.0;
                hits.push(ChunkHit {
                    chunk_id: chunk_ids
                        .map(|a| a.value(i).to_vec())
                        .unwrap_or_default(),
                    file_path: file_paths
                        .map(|a| a.value(i).to_string())
                        .unwrap_or_default(),
                    symbol_qualified: symbol_qualified
                        .map(|a| a.value(i).to_string())
                        .unwrap_or_default(),
                    start_line: start_lines.map(|a| a.value(i)).unwrap_or(0),
                    end_line: end_lines.map(|a| a.value(i)).unwrap_or(0),
                    content: contents
                        .map(|a| a.value(i).to_string())
                        .unwrap_or_default(),
                    turbo_code: turbo_codes
                        .map(|a| a.value(i).to_vec())
                        .unwrap_or_default(),
                    score,
                });
            }
        }
        Ok(hits)
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }
}

fn glob_to_like(glob: &str) -> Option<String> {
    let mut pattern = String::new();
    for ch in glob.chars() {
        match ch {
            '*' => pattern.push('%'),
            '?' => pattern.push('_'),
            '\'' | '\\' | '%' | '_' => return None,
            _ => pattern.push(ch),
        }
    }
    Some(pattern)
}

#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub chunk_id: Vec<u8>,
    pub file_path: String,
    pub symbol_qualified: String,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
    pub turbo_code: Vec<u8>,
    pub score: f32,
}
