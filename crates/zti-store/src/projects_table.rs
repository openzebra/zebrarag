use anyhow::Result;
use arrow::array::RecordBatch;
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

    pub async fn upsert(&self, batch: RecordBatch) -> Result<()> {
        self.table.add(batch).execute().await?;
        Ok(())
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }
}
