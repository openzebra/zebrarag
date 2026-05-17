use anyhow::Result;
use arrow::array::{
    Array, BinaryArray, FixedSizeBinaryArray, Float32Array, ListArray, RecordBatch,
    RecordBatchIterator, StringArray, UInt32Array,
};
use lancedb::index::Index;
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use std::fmt::Write as _;
use std::sync::Arc;

use crate::schema;

pub struct ChunksTable {
    table: Table,
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
        let row_count = self.table.count_rows(None).await?;
        if row_count == 0 {
            return Ok(());
        }
        if row_count < 256 {
            tracing::warn!(
                "skipping IVF-PQ index: need >= 256 rows, got {row_count}"
            );
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

        if let Some(langs) = languages
            && !langs.is_empty()
        {
            let list = langs
                .iter()
                .map(|l| format!("'{}'", l))
                .collect::<Vec<_>>()
                .join(",");
            filters.push(format!("language IN ({})", list));
        }

        if let Some(glob) = path_glob
            && let Some(pattern) = glob_to_like(glob)
        {
            filters.push(format!("file_path LIKE '{}'", pattern));
        }

        if !filters.is_empty() {
            let combined = filters.join(" AND ");
            q = q.only_if(combined);
        }

        let results = q.execute().await?;

        let mut hits = Vec::with_capacity(k);
        let mut stream = std::pin::pin!(results);
        use futures::StreamExt;
        while let Some(batch) = stream.next().await {
            decode_batch(&batch?, true, &mut hits);
        }
        Ok(hits)
    }

    pub async fn get_by_sym_ids(&self, ids: &[u32]) -> Result<Vec<ChunkHit>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut filter = String::with_capacity(16 + ids.len() * 6);
        filter.push_str("sym_id IN (");
        for (i, id) in ids.iter().enumerate() {
            if i > 0 {
                filter.push(',');
            }
            let _ = write!(filter, "{}", id);
        }
        filter.push(')');

        let results = self.table.query().only_if(filter).execute().await?;

        let mut hits = Vec::with_capacity(ids.len());
        let mut stream = std::pin::pin!(results);
        use futures::StreamExt;
        while let Some(batch) = stream.next().await {
            decode_batch(&batch?, false, &mut hits);
        }
        Ok(hits)
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }
}

fn decode_batch(batch: &RecordBatch, has_distance: bool, out: &mut Vec<ChunkHit>) {
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
    let symbol_kinds = batch
        .column_by_name("symbol_kind")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let sym_ids = batch
        .column_by_name("sym_id")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let parent_sym_ids = batch
        .column_by_name("parent_sym_id")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let appendix_sym_ids = batch
        .column_by_name("appendix_sym_ids")
        .and_then(|c| c.as_any().downcast_ref::<ListArray>());
    let chunk_ids = batch
        .column_by_name("chunk_id")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>());
    let turbo_codes = batch
        .column_by_name("turbo_code")
        .and_then(|c| c.as_any().downcast_ref::<BinaryArray>());
    let distances = if has_distance {
        batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
    } else {
        None
    };

    out.reserve(num_rows);
    for i in 0..num_rows {
        let d = distances.map(|a| a.value(i)).unwrap_or(0.0);
        let score = 1.0 - d / 2.0;
        let appendix = appendix_sym_ids
            .and_then(|arr| {
                if arr.is_null(i) {
                    return None;
                }
                let inner = arr.value(i);
                let u32_inner = inner.as_any().downcast_ref::<UInt32Array>()?;
                let mut v = Vec::with_capacity(u32_inner.len());
                for j in 0..u32_inner.len() {
                    if !u32_inner.is_null(j) {
                        v.push(u32_inner.value(j));
                    }
                }
                Some(v)
            })
            .unwrap_or_default();
        let parent_sym_id = parent_sym_ids.and_then(|a| {
            if a.is_null(i) { None } else { Some(a.value(i)) }
        });
        out.push(ChunkHit {
            chunk_id: chunk_ids
                .map(|a| a.value(i).to_vec())
                .unwrap_or_default(),
            file_path: file_paths
                .map(|a| a.value(i).to_string())
                .unwrap_or_default(),
            symbol_qualified: symbol_qualified
                .map(|a| a.value(i).to_string())
                .unwrap_or_default(),
            symbol_kind: symbol_kinds
                .map(|a| a.value(i).to_string())
                .unwrap_or_default(),
            sym_id: sym_ids.map(|a| a.value(i)).unwrap_or(0),
            parent_sym_id,
            appendix_sym_ids: appendix,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_to_like_translates_wildcards() {
        assert_eq!(glob_to_like("src/**").as_deref(), Some("src/%%"));
        assert_eq!(glob_to_like("a?b").as_deref(), Some("a_b"));
        assert_eq!(glob_to_like("plain").as_deref(), Some("plain"));
    }

    #[test]
    fn glob_to_like_rejects_sql_injection() {
        // Single quote, backslash, and raw SQL wildcards are not user-safe.
        assert!(glob_to_like("' OR 1=1 --").is_none());
        assert!(glob_to_like(r"back\slash").is_none());
        assert!(glob_to_like("raw_underscore").is_none());
        assert!(glob_to_like("raw%percent").is_none());
    }
}

#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub chunk_id: Vec<u8>,
    pub file_path: String,
    pub symbol_qualified: String,
    pub symbol_kind: String,
    pub sym_id: u32,
    pub parent_sym_id: Option<u32>,
    pub appendix_sym_ids: Vec<u32>,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
    pub turbo_code: Vec<u8>,
    pub score: f32,
}
