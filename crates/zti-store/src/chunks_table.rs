use std::borrow::Cow;
use std::fmt::Write as _;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::array::{
    Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, ListArray,
    RecordBatch, StringArray, UInt8Array, UInt32Array,
};
use arrow::datatypes::DataType;
use futures::StreamExt;
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::vector::{
    IvfHnswPqIndexBuilder, IvfHnswSqIndexBuilder, IvfPqIndexBuilder, IvfRqIndexBuilder,
    IvfSqIndexBuilder,
};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;

use zti_common::chunk_strategy::ChunkStrategy;
use zti_common::file_type::FileType;

use crate::schema;

/// Columns `decode_batch` reads when `has_distance = true`.
/// Excludes `language`, `indexed_at_ns`, and the heavy `embedding` column
/// (3 KB/row at dim=768) that `decode_batch` never touches.
const CHUNK_HIT_COLS_WITH_DISTANCE: &[&str] = &[
    "chunk_id",
    "file_path",
    "file_type",
    "symbol_qualified",
    "symbol_kind",
    "sym_id",
    "sub_chunk_idx",
    "total_sub_chunks",
    "chunk_strategy",
    "parent_sym_id",
    "appendix_sym_ids",
    "start_line",
    "end_line",
    "content",
    "turbo_code",
    "_distance",
];

/// Same as [`CHUNK_HIT_COLS_WITH_DISTANCE`] but without `_distance` — used by
/// non-vector queries (`fetch_by_chunk_ids`, `lexical_match`, `get_by_sym_ids`).
const CHUNK_HIT_COLS_NO_DISTANCE: &[&str] = &[
    "chunk_id",
    "file_path",
    "file_type",
    "symbol_qualified",
    "symbol_kind",
    "sym_id",
    "sub_chunk_idx",
    "total_sub_chunks",
    "chunk_strategy",
    "parent_sym_id",
    "appendix_sym_ids",
    "start_line",
    "end_line",
    "content",
    "turbo_code",
];

pub struct ChunksTable {
    table: Table,
    index_created: bool,
}

impl ChunksTable {
    pub async fn open(db: &lancedb::Connection, dim: usize) -> Result<Self> {
        let name = "chunks";
        let table = if db.table_names().execute().await?.iter().any(|n| n == name) {
            let existing = db.open_table(name).execute().await?;
            let existing_schema = existing.schema().await?;
            let existing_dim = existing_schema
                .field_with_name("embedding")
                .ok()
                .and_then(|f| match f.data_type() {
                    DataType::FixedSizeList(_, n) => Some(*n as usize),
                    _ => None,
                })
                .unwrap_or(0);

            let has_new_cols = existing_schema.field_with_name("sub_chunk_idx").is_ok()
                && existing_schema.field_with_name("file_type").is_ok();

            if existing_dim != dim || !has_new_cols {
                tracing::warn!(
                    "schema changed (dim={}, has_new_cols={}), recreating chunks table",
                    existing_dim,
                    has_new_cols,
                );
                db.drop_table("files", &[]).await.ok();
                db.drop_table(name, &[]).await?;
                let schema = Arc::new(schema::chunks_schema(dim));
                db.create_empty_table(name, schema).execute().await?
            } else {
                existing
            }
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
        crate::upsert::upsert_batch(&self.table, "chunk_id", batch).await
    }

    /// Append already-deduplicated chunk batches in one commit. Callers must
    /// ensure the incoming `chunk_id`s are absent from the table (the indexer
    /// deletes a file's old chunks before re-inserting). See
    /// [`crate::upsert::append_batches`].
    pub async fn append_batches(&self, batches: Vec<RecordBatch>) -> Result<()> {
        crate::upsert::append_batches(&self.table, batches).await
    }

    pub async fn build_index(&mut self, params: &zti_ann::SearchParams) -> Result<()> {
        if self.index_created {
            return Ok(());
        }
        let n = self.table.count_rows(None).await?;
        if n == 0 || !params.method.is_lancedb_index() {
            self.index_created = true;
            return Ok(());
        }
        if n < 256 {
            tracing::warn!("skipping ANN index: need >= 256 rows, got {n}");
            self.index_created = true;
            return Ok(());
        }

        let dist = lancedb::DistanceType::Cosine;
        let np = params.num_partitions;
        let m = params.m;
        let efc = params.ef_construction;
        let nsv = params.num_sub_vectors;

        let index = match params.method {
            zti_ann::SearchMethod::IvfHnswSq => Index::IvfHnswSq(
                IvfHnswSqIndexBuilder::default()
                    .distance_type(dist)
                    .num_partitions(np)
                    .num_edges(m)
                    .ef_construction(efc),
            ),
            zti_ann::SearchMethod::IvfHnswPq => Index::IvfHnswPq(
                IvfHnswPqIndexBuilder::default()
                    .distance_type(dist)
                    .num_partitions(np)
                    .num_edges(m)
                    .ef_construction(efc)
                    .num_sub_vectors(nsv),
            ),
            zti_ann::SearchMethod::IvfPq => Index::IvfPq(
                IvfPqIndexBuilder::default()
                    .distance_type(dist)
                    .num_partitions(np)
                    .num_sub_vectors(nsv),
            ),
            zti_ann::SearchMethod::IvfSq => Index::IvfSq(
                IvfSqIndexBuilder::default()
                    .distance_type(dist)
                    .num_partitions(np),
            ),
            zti_ann::SearchMethod::IvfRq => Index::IvfRq(
                IvfRqIndexBuilder::default()
                    .distance_type(dist)
                    .num_partitions(np),
            ),
            zti_ann::SearchMethod::Flat
            | zti_ann::SearchMethod::Usearch
            | zti_ann::SearchMethod::TurboQuant => {
                self.index_created = true;
                return Ok(());
            }
        };

        self.table
            .create_index(&["embedding"], index)
            .execute()
            .await?;
        self.index_created = true;
        Ok(())
    }

    pub async fn ensure_fts_indexes(&self) -> Result<()> {
        let params = || {
            FtsIndexBuilder::default()
                .base_tokenizer("simple".to_string())
                .lower_case(true)
                .ascii_folding(true)
                .stem(false)
                .remove_stop_words(false)
                .with_position(false)
        };
        self.table
            .create_index(&["content"], Index::FTS(params()))
            .execute()
            .await?;
        self.table
            .create_index(&["symbol_qualified"], Index::FTS(params()))
            .execute()
            .await?;
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
        params: &zti_ann::SearchParams,
        languages: Option<&[String]>,
        path_glob: Option<&str>,
        include_tests: bool,
    ) -> Result<Vec<ChunkHit>> {
        let mut q = self
            .table
            .query()
            .nearest_to(query)?
            .distance_type(lancedb::DistanceType::Cosine)
            .limit(k)
            .select(lancedb::query::Select::columns(
                CHUNK_HIT_COLS_WITH_DISTANCE,
            ));

        if params.method.is_lancedb_index() {
            q = q
                .nprobes(params.nprobes as usize)
                .refine_factor(params.refine_factor);
            if matches!(
                params.method,
                zti_ann::SearchMethod::IvfHnswSq | zti_ann::SearchMethod::IvfHnswPq
            ) {
                q = q.ef(params.ef_search as usize);
            }
        }

        if let Some(filter) = build_lang_path_filter(languages, path_glob, include_tests) {
            q = q.only_if(filter);
        }

        let results = q.execute().await?;

        let mut hits = Vec::with_capacity(k);
        let mut stream = std::pin::pin!(results);
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

        let results = self
            .table
            .query()
            .only_if(filter)
            .select(lancedb::query::Select::columns(CHUNK_HIT_COLS_NO_DISTANCE))
            .execute()
            .await?;

        let mut hits = Vec::with_capacity(ids.len());
        let mut stream = std::pin::pin!(results);
        while let Some(batch) = stream.next().await {
            decode_batch(&batch?, false, &mut hits);
        }
        Ok(hits)
    }

    pub async fn fetch_by_chunk_ids(
        &self,
        ids: &[[u8; 16]],
        languages: Option<&[String]>,
        path_glob: Option<&str>,
        include_tests: bool,
    ) -> Result<Vec<ChunkHit>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let hex_parts: Vec<String> = ids
            .iter()
            .map(|id| {
                let hex: String = id.iter().map(|b| format!("{:02x}", b)).collect();
                format!("X'{}'", hex)
            })
            .collect();

        let chunk_filter = format!("chunk_id IN ({})", hex_parts.join(","));
        let filter = match build_lang_path_filter(languages, path_glob, include_tests) {
            Some(lp) => format!("{} AND {}", chunk_filter, lp),
            None => chunk_filter,
        };

        let results = self
            .table
            .query()
            .only_if(filter)
            .select(lancedb::query::Select::columns(CHUNK_HIT_COLS_NO_DISTANCE))
            .execute()
            .await?;

        let mut hits = Vec::with_capacity(ids.len());
        let mut stream = std::pin::pin!(results);
        while let Some(batch) = stream.next().await {
            decode_batch(&batch?, false, &mut hits);
        }
        Ok(hits)
    }

    pub async fn iter_vectors<F>(&self, mut on_row: F) -> Result<usize>
    where
        F: FnMut(&[u8; 16], &[f32]),
    {
        let results = self
            .table
            .query()
            .select(lancedb::query::Select::columns(&["chunk_id", "embedding"]))
            .execute()
            .await?;

        let mut stream = std::pin::pin!(results);
        let mut count = 0usize;
        while let Some(batch) = stream.next().await {
            let b = batch?;

            let ids = b
                .column_by_name("chunk_id")
                .ok_or_else(|| anyhow!("missing column 'chunk_id'"))?
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| anyhow!("chunk_id is not FixedSizeBinary"))?;

            let embs = b
                .column_by_name("embedding")
                .ok_or_else(|| anyhow!("missing column 'embedding'"))?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| anyhow!("embedding is not FixedSizeList"))?;

            let values = embs
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("embedding values not Float32"))?
                .values();
            let dim = embs.value_length() as usize;

            for i in 0..b.num_rows() {
                let raw = ids.value(i);
                let id: &[u8; 16] = raw
                    .try_into()
                    .map_err(|_| anyhow!("chunk_id length != 16"))?;
                let start = i * dim;
                on_row(id, &values[start..start + dim]);
                count += 1;
            }
        }
        Ok(count)
    }

    pub async fn iter_turbo_codes<F>(
        &self,
        languages: Option<&[String]>,
        path_glob: Option<&str>,
        include_tests: bool,
        mut on_row: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8; 16], &[u8]) -> Result<bool>,
    {
        let mut filter = String::from("turbo_code IS NOT NULL");
        if let Some(lp) = build_lang_path_filter(languages, path_glob, include_tests) {
            filter.push_str(" AND ");
            filter.push_str(&lp);
        }

        let results = self
            .table
            .query()
            .select(lancedb::query::Select::columns(&["chunk_id", "turbo_code"]))
            .only_if(filter)
            .execute()
            .await?;

        let mut stream = std::pin::pin!(results);
        let mut count = 0usize;
        while let Some(batch) = stream.next().await {
            let b = batch?;
            let ids = b
                .column_by_name("chunk_id")
                .ok_or_else(|| anyhow!("missing column 'chunk_id'"))?
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| anyhow!("chunk_id is not FixedSizeBinary"))?;
            let codes = b
                .column_by_name("turbo_code")
                .ok_or_else(|| anyhow!("missing column 'turbo_code'"))?
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| anyhow!("turbo_code is not Binary"))?;

            for i in 0..b.num_rows() {
                if codes.is_null(i) {
                    continue;
                }
                let raw = ids.value(i);
                let id: &[u8; 16] = match raw.try_into() {
                    Ok(id) => id,
                    Err(_) => continue,
                };
                if !on_row(id, codes.value(i))? {
                    return Ok(count);
                }
                count += 1;
            }
        }
        Ok(count)
    }

    pub async fn knn_exhaustive(
        &self,
        query: &[f32],
        k: usize,
        languages: Option<&[String]>,
        path_glob: Option<&str>,
        include_tests: bool,
    ) -> Result<Vec<ChunkHit>> {
        let mut q = self
            .table
            .query()
            .nearest_to(query)?
            .distance_type(lancedb::DistanceType::Cosine)
            .bypass_vector_index()
            .limit(k)
            .select(lancedb::query::Select::columns(
                CHUNK_HIT_COLS_WITH_DISTANCE,
            ));

        if let Some(filter) = build_lang_path_filter(languages, path_glob, include_tests) {
            q = q.only_if(filter);
        }

        let results = q.execute().await?;

        let mut hits = Vec::with_capacity(k);
        let mut stream = std::pin::pin!(results);
        while let Some(batch) = stream.next().await {
            decode_batch(&batch?, true, &mut hits);
        }
        Ok(hits)
    }

    /// BM25 lexical candidate fetch using LanceDB full-text search. The stream
    /// order is the BM25 rank order; RRF consumes only the ordinal rank.
    pub async fn lexical_match(
        &self,
        query: &str,
        languages: Option<&[String]>,
        path_glob: Option<&str>,
        include_tests: bool,
        k: usize,
    ) -> Result<Vec<ChunkHit>> {
        if query.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let fts_columns = ["content".to_string(), "symbol_qualified".to_string()];
        let fts_query = FullTextSearchQuery::new(query.to_string()).with_columns(&fts_columns)?;
        let mut q = self
            .table
            .query()
            .full_text_search(fts_query)
            .limit(k)
            .select(lancedb::query::Select::columns(CHUNK_HIT_COLS_NO_DISTANCE));
        if let Some(pred) = build_lang_path_filter(languages, path_glob, include_tests) {
            q = q.only_if(pred);
        }
        let mut hits = Vec::with_capacity(k);
        let mut stream = std::pin::pin!(q.execute().await?);
        while let Some(batch) = stream.next().await {
            decode_batch(&batch?, false, &mut hits);
        }
        Ok(hits)
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }

    pub async fn optimize(&self) -> Result<()> {
        use lancedb::table::OptimizeAction;

        self.table.optimize(OptimizeAction::All).await?;
        Ok(())
    }
}

fn build_lang_path_filter(
    languages: Option<&[String]>,
    path_glob: Option<&str>,
    include_tests: bool,
) -> Option<String> {
    let mut filters: Vec<Cow<'static, str>> = Vec::with_capacity(3);
    if !include_tests {
        // `FileType::Test` is persisted as `1`; keep this borrowed so default
        // source searches do not allocate for the test-hiding predicate.
        filters.push(Cow::Borrowed("file_type != 1"));
    }
    if let Some(langs) = languages.filter(|langs| !langs.is_empty()) {
        let list = langs
            .iter()
            .map(|lang| format!("'{lang}'"))
            .collect::<Vec<_>>()
            .join(",");
        filters.push(Cow::Owned(format!("language IN ({list})")));
    }
    if let Some(pattern) = path_glob.and_then(glob_to_like) {
        filters.push(Cow::Owned(format!("file_path LIKE '{pattern}'")));
    }
    (!filters.is_empty()).then(|| filters.join(" AND "))
}

fn decode_batch(batch: &RecordBatch, has_distance: bool, out: &mut Vec<ChunkHit>) {
    let num_rows = batch.num_rows();
    let file_paths = batch
        .column_by_name("file_path")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let file_types = batch
        .column_by_name("file_type")
        .and_then(|c| c.as_any().downcast_ref::<UInt8Array>());
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
    let sub_chunk_idxs = batch
        .column_by_name("sub_chunk_idx")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let total_sub_chunks_arr = batch
        .column_by_name("total_sub_chunks")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
    let chunk_strategies = batch
        .column_by_name("chunk_strategy")
        .and_then(|c| c.as_any().downcast_ref::<arrow::array::UInt8Array>());
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
        let score = 1.0 - d;
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
        let parent_sym_id =
            parent_sym_ids.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) });
        let cid: [u8; 16] = chunk_ids
            .and_then(|a| a.value(i).try_into().ok())
            .unwrap_or([0u8; 16]);
        let file_type = file_types
            .and_then(|array| array.is_valid(i).then(|| array.value(i)))
            .and_then(|value| FileType::try_from(value).ok())
            .unwrap_or_default();
        out.push(ChunkHit {
            chunk_id: cid,
            file_path: file_paths
                .map(|a| a.value(i).to_string())
                .unwrap_or_default(),
            file_type,
            symbol_qualified: symbol_qualified
                .map(|a| a.value(i).to_string())
                .unwrap_or_default(),
            symbol_kind: symbol_kinds
                .map(|a| a.value(i).to_string())
                .unwrap_or_default(),
            sym_id: sym_ids.map(|a| a.value(i)).unwrap_or(0),
            sub_chunk_idx: sub_chunk_idxs.map(|a| a.value(i)).unwrap_or(0),
            total_sub_chunks: total_sub_chunks_arr.map(|a| a.value(i)).unwrap_or(1),
            chunk_strategy: chunk_strategies
                .map(|a| ChunkStrategy::from(a.value(i)))
                .unwrap_or(ChunkStrategy::Symbol),
            parent_sym_id,
            appendix_sym_ids: appendix,
            start_line: start_lines.map(|a| a.value(i)).unwrap_or(0),
            end_line: end_lines.map(|a| a.value(i)).unwrap_or(0),
            content: contents.map(|a| a.value(i).to_string()).unwrap_or_default(),
            turbo_code: turbo_codes.map(|a| a.value(i).to_vec()).unwrap_or_default(),
            score,
        });
    }
}

fn glob_to_like(glob: &str) -> Option<String> {
    let mut pattern = String::with_capacity(glob.len());
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
    use arrow::array::UInt8Array;
    use arrow::datatypes::{DataType, Field, Schema};

    #[test]
    fn glob_to_like_translates_wildcards() {
        assert_eq!(glob_to_like("src/**").as_deref(), Some("src/%%"));
        assert_eq!(glob_to_like("a?b").as_deref(), Some("a_b"));
        assert_eq!(glob_to_like("plain").as_deref(), Some("plain"));
    }

    #[test]
    fn glob_to_like_rejects_sql_injection() {
        assert!(glob_to_like("' OR 1=1 --").is_none());
        assert!(glob_to_like(r"back\slash").is_none());
        assert!(glob_to_like("raw_underscore").is_none());
        assert!(glob_to_like("raw%percent").is_none());
    }

    #[test]
    fn build_filter_hides_tests_by_default() {
        assert_eq!(
            build_lang_path_filter(None, None, false).as_deref(),
            Some("file_type != 1"),
        );
        assert_eq!(build_lang_path_filter(None, None, true), None);
    }

    #[test]
    fn build_filter_combines_scope_predicates() {
        let languages = vec!["rust".to_string(), "solidity".to_string()];
        assert_eq!(
            build_lang_path_filter(Some(&languages), Some("src/**"), false).as_deref(),
            Some("file_type != 1 AND language IN ('rust','solidity') AND file_path LIKE 'src/%%'"),
        );
    }

    #[test]
    fn test_decode_batch_recursive_fields() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sym_id", DataType::UInt32, false),
            Field::new("sub_chunk_idx", DataType::UInt32, false),
            Field::new("total_sub_chunks", DataType::UInt32, false),
            Field::new("chunk_strategy", DataType::UInt8, false),
        ]));

        let sym_ids = Arc::new(UInt32Array::from(vec![100u32, 101, 102]));
        let sub_chunk_idxs = Arc::new(UInt32Array::from(vec![0u32, 0, 1]));
        let total_sub_chunks_arr = Arc::new(UInt32Array::from(vec![1u32, 2, 2]));
        let chunk_strategies = Arc::new(UInt8Array::from(vec![0u8, 1, 255]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                sym_ids,
                sub_chunk_idxs,
                total_sub_chunks_arr,
                chunk_strategies,
            ],
        )
        .unwrap();

        let mut hits = Vec::new();
        decode_batch(&batch, false, &mut hits);

        assert_eq!(hits.len(), 3);

        assert_eq!(hits[0].sym_id, 100);
        assert_eq!(hits[0].sub_chunk_idx, 0);
        assert_eq!(hits[0].total_sub_chunks, 1);
        assert_eq!(hits[0].chunk_strategy, ChunkStrategy::Symbol);

        assert_eq!(hits[1].sym_id, 101);
        assert_eq!(hits[1].sub_chunk_idx, 0);
        assert_eq!(hits[1].total_sub_chunks, 2);
        assert_eq!(hits[1].chunk_strategy, ChunkStrategy::Recursive);

        assert_eq!(hits[2].sym_id, 102);
        assert_eq!(hits[2].sub_chunk_idx, 1);
        assert_eq!(hits[2].total_sub_chunks, 2);
        assert_eq!(hits[2].chunk_strategy, ChunkStrategy::Symbol);
    }
}

#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub chunk_id: [u8; 16],
    pub file_path: String,
    pub file_type: FileType,
    pub symbol_qualified: String,
    pub symbol_kind: String,
    pub sym_id: u32,
    pub sub_chunk_idx: u32,
    pub total_sub_chunks: u32,
    pub chunk_strategy: ChunkStrategy,
    pub parent_sym_id: Option<u32>,
    pub appendix_sym_ids: Vec<u32>,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
    pub turbo_code: Vec<u8>,
    pub score: f32,
}
