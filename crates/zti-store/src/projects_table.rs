use anyhow::Result;
use arrow::array::{RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use std::sync::Arc;

use crate::schema;

pub struct ProjectsTable {
    table: Table,
}

impl ProjectsTable {
    pub async fn open(db: &lancedb::Connection) -> Result<Self> {
        let name = "projects";
        let table = if db.table_names().execute().await?.contains(&name.to_string()) {
            db.open_table(name).execute().await?
        } else {
            let schema = Arc::new(schema::projects_schema());
            db.create_empty_table(name, schema).execute().await?
        };
        Ok(Self { table })
    }

    pub async fn get(&self, project_id: &[u8; 32]) -> Result<Option<ProjectRow>> {
        let hex_id: String = project_id.iter().map(|b| format!("{:02x}", b)).collect();
        let filter = format!("project_id = '\\x{}'", hex_id);
        let results = self.table.query().only_if(filter).execute().await?;

        let mut stream = std::pin::pin!(results);
        use futures::StreamExt;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }

            let root_paths = batch
                .column_by_name("root_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let model_ids = batch
                .column_by_name("model_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let model_dims = batch
                .column_by_name("model_dim")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let total_chunks = batch
                .column_by_name("total_chunks")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
            let total_files = batch
                .column_by_name("total_files")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
            let last_indexed = batch
                .column_by_name("last_indexed_ns")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
            let created_at = batch
                .column_by_name("created_at_ns")
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());

            let i = 0;
            return Ok(Some(ProjectRow {
                project_id: project_id.to_vec(),
                root_path: root_paths
                    .map(|a| a.value(i).to_string())
                    .unwrap_or_default(),
                model_id: model_ids
                    .map(|a| a.value(i).to_string())
                    .unwrap_or_default(),
                model_dim: model_dims.map(|a| a.value(i)).unwrap_or(0),
                total_chunks: total_chunks.map(|a| a.value(i)).unwrap_or(0),
                total_files: total_files.map(|a| a.value(i)).unwrap_or(0),
                last_indexed_ns: last_indexed.map(|a| a.value(i)).unwrap_or(0),
                created_at_ns: created_at.map(|a| a.value(i)).unwrap_or(0),
            }));
        }
        Ok(None)
    }

    pub async fn upsert(&self, batch: RecordBatch) -> Result<()> {
        let schema = batch.schema();
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

        let mut builder = self.table.merge_insert(&["project_id"]);
        builder.when_matched_update_all(None);
        builder.when_not_matched_insert_all();
        builder.execute(reader).await?;

        Ok(())
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }
}

#[derive(Debug, Clone)]
pub struct ProjectRow {
    pub project_id: Vec<u8>,
    pub root_path: String,
    pub model_id: String,
    pub model_dim: u32,
    pub total_chunks: u64,
    pub total_files: u64,
    pub last_indexed_ns: u64,
    pub created_at_ns: u64,
}
