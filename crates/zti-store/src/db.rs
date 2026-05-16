use std::path::PathBuf;

use anyhow::Result;
use lancedb::connect;

use crate::chunks_table::ChunksTable;
use crate::files_table::FilesTable;
use crate::projects_table::ProjectsTable;

#[derive(Clone)]
pub struct Db {
    db: lancedb::Connection,
    root: PathBuf,
}

impl Db {
    pub async fn open(project_id: &[u8; 32]) -> Result<Self> {
        let root = zti_common::paths::project_dir(project_id)?;
        let lance_dir = root.join("lance");
        std::fs::create_dir_all(&lance_dir)?;

        let db = connect(lance_dir.to_str().ok_or_else(|| anyhow::anyhow!("invalid path"))?).execute().await?;

        Ok(Self { db, root: lance_dir })
    }

    pub async fn open_global() -> Result<Self> {
        let data = zti_common::paths::data_dir()?;
        let registry = data.join("registry");
        std::fs::create_dir_all(&registry)?;

        let db = connect(registry.to_str().ok_or_else(|| anyhow::anyhow!("invalid path"))?).execute().await?;

        Ok(Self {
            db,
            root: registry,
        })
    }

    pub fn connection(&self) -> &lancedb::Connection {
        &self.db
    }

    pub async fn chunks_table(&self, dim: usize) -> Result<ChunksTable> {
        ChunksTable::open(&self.db, dim).await
    }

    pub async fn files_table(&self) -> Result<FilesTable> {
        FilesTable::open(&self.db).await
    }

    pub async fn projects_table(&self) -> Result<ProjectsTable> {
        ProjectsTable::open(&self.db).await
    }
}
