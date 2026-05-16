use anyhow::Result;
use arrow::array::{Float32Array, RecordBatch, StringArray, UInt32Array};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use std::sync::Arc;

use crate::schema;

pub struct ChunksTable {
    table: Table,
    dim: usize,
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
        Ok(Self { table, dim })
    }

    pub async fn upsert(&self, batch: RecordBatch) -> Result<()> {
        self.table.add(batch).execute().await?;
        Ok(())
    }

    pub async fn knn(&self, query: &[f32], k: usize) -> Result<Vec<ChunkHit>> {
        let results = self.table.query()
            .nearest_to(query)?
            .limit(k)
            .execute()
            .await?;

        let mut hits = Vec::new();
        let mut stream = std::pin::pin!(results);
        use futures::StreamExt;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let num_rows = batch.num_rows();
            let file_paths = batch.column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let start_lines = batch.column_by_name("start_line")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let end_lines = batch.column_by_name("end_line")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let contents = batch.column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let symbol_qualified = batch.column_by_name("symbol_qualified")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let distances = batch.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..num_rows {
                hits.push(ChunkHit {
                    file_path: file_paths.map(|a| a.value(i).to_string()).unwrap_or_default(),
                    symbol_qualified: symbol_qualified.map(|a| a.value(i).to_string()).unwrap_or_default(),
                    start_line: start_lines.map(|a| a.value(i)).unwrap_or(0),
                    end_line: end_lines.map(|a| a.value(i)).unwrap_or(0),
                    content: contents.map(|a| a.value(i).to_string()).unwrap_or_default(),
                    score: distances.map(|a| 1.0 - a.value(i)).unwrap_or(0.0),
                });
            }
        }
        Ok(hits)
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }
}

#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub file_path: String,
    pub symbol_qualified: String,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
    pub score: f32,
}
