use std::fmt::Write as _;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::array::{
    Array, BinaryArray, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, ListArray,
    RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
};
use arrow::datatypes::DataType;
use futures::StreamExt;
use lancedb::index::Index;
use lancedb::index::vector::{
    IvfHnswPqIndexBuilder, IvfHnswSqIndexBuilder, IvfPqIndexBuilder, IvfRqIndexBuilder,
    IvfSqIndexBuilder,
};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;

use crate::schema;

pub struct ChunksTable {
    table: Table,
    index_created: bool,
}

impl ChunksTable {
    pub async fn open(db: &lancedb::Connection, dim: usize) -> Result<Self> {
        let name = "chunks";
        let table = if db.table_names().execute().await?.iter().any(|n| n == name) {
            let existing = db.open_table(name).execute().await?;
            let existing_dim = existing
                .schema()
                .await?
                .field_with_name("embedding")
                .ok()
                .and_then(|f| match f.data_type() {
                    DataType::FixedSizeList(_, n) => Some(*n as usize),
                    _ => None,
                })
                .unwrap_or(0);

            if existing_dim != dim {
                tracing::warn!(
                    "embedding dim changed ({} → {}), recreating chunks table",
                    existing_dim,
                    dim
                );
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
        let schema = batch.schema();
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

        let mut builder = self.table.merge_insert(&["chunk_id"]);
        builder.when_matched_update_all(None);
        builder.when_not_matched_insert_all();
        builder.execute(reader).await?;

        Ok(())
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
    ) -> Result<Vec<ChunkHit>> {
        let mut q = self
            .table
            .query()
            .nearest_to(query)?
            .distance_type(lancedb::DistanceType::Cosine)
            .limit(k);

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

        if let Some(filter) = build_lang_path_filter(languages, path_glob) {
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

        let results = self.table.query().only_if(filter).execute().await?;

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
        let filter = match build_lang_path_filter(languages, path_glob) {
            Some(lp) => format!("{} AND {}", chunk_filter, lp),
            None => chunk_filter,
        };

        let results = self.table.query().only_if(filter).execute().await?;

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
        mut on_row: F,
    ) -> Result<usize>
    where
        F: FnMut(&[u8; 16], &[u8]),
    {
        let mut filter = String::from("turbo_code IS NOT NULL");
        if let Some(lp) = build_lang_path_filter(languages, path_glob) {
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
                on_row(id, codes.value(i));
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
    ) -> Result<Vec<ChunkHit>> {
        let mut q = self
            .table
            .query()
            .nearest_to(query)?
            .distance_type(lancedb::DistanceType::Cosine)
            .bypass_vector_index()
            .limit(k);

        if let Some(filter) = build_lang_path_filter(languages, path_glob) {
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

    /// Lexical candidate fetch. For each entry in `words` (lowercased,
    /// alphanumeric+`_`, len ≥ 3) OR-matches `LOWER(symbol_qualified) LIKE
    /// '%w%'` and `LOWER(content) LIKE '%w%'`. Words that contain SQL
    /// specials (`'`, `\`, `%`, `_`) are dropped — same policy as
    /// [`glob_to_like`] — so no escape-handling code is duplicated here.
    /// The language/path filter, if any, is AND-ed in via
    /// [`build_lang_path_filter`].
    ///
    /// Used by `zti_pipeline::search::search` as the lexical leg of a hybrid
    /// retrieval. Returns at most `k` rows. Score field on each `ChunkHit`
    /// is set to `1.0` (no distance column on a non-vector query); the
    /// keyword boost stage adds the per-word lift on top of that.
    pub async fn lexical_match(
        &self,
        words: &[&str],
        languages: Option<&[String]>,
        path_glob: Option<&str>,
        k: usize,
    ) -> Result<Vec<ChunkHit>> {
        if words.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let mut clauses = String::with_capacity(words.len() * 80);
        let mut emitted: usize = 0;
        for w in words {
            if !word_is_sql_safe(w) {
                continue;
            }
            if emitted > 0 {
                clauses.push_str(" OR ");
            }
            let _ = write!(
                clauses,
                "(LOWER(symbol_qualified) LIKE '%{w}%' OR LOWER(content) LIKE '%{w}%')",
            );
            emitted += 1;
        }
        if emitted == 0 {
            return Ok(Vec::new());
        }

        let predicate = match build_lang_path_filter(languages, path_glob) {
            Some(lp) => {
                let mut s = String::with_capacity(lp.len() + clauses.len() + 16);
                s.push_str(&lp);
                s.push_str(" AND (");
                s.push_str(&clauses);
                s.push(')');
                s
            }
            None => clauses,
        };

        let results = self
            .table
            .query()
            .only_if(predicate)
            .limit(k)
            .execute()
            .await?;

        let mut hits = Vec::with_capacity(k);
        let mut stream = std::pin::pin!(results);
        while let Some(batch) = stream.next().await {
            decode_batch(&batch?, false, &mut hits);
        }
        // `decode_batch(.., has_distance=false, ..)` initialised `score = 1.0`
        // for every row (1 - 0.0). That gives lexical-only hits a uniform
        // baseline; the keyword boost stage lifts the ones whose query words
        // actually hit name vs body.
        Ok(hits)
    }

    pub async fn len(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await?)
    }
}

fn build_lang_path_filter(languages: Option<&[String]>, path_glob: Option<&str>) -> Option<String> {
    let mut filters = Vec::with_capacity(2);
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
    if filters.is_empty() {
        None
    } else {
        Some(filters.join(" AND "))
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
        out.push(ChunkHit {
            chunk_id: cid,
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
            content: contents.map(|a| a.value(i).to_string()).unwrap_or_default(),
            turbo_code: turbo_codes.map(|a| a.value(i).to_vec()).unwrap_or_default(),
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

/// `true` iff `word` is safe to embed inside a SQL `LIKE '%…%'` predicate
/// without further escaping. Mirrors [`glob_to_like`]'s reject-set so the
/// caller never has to think about SQL escaping. Callers tokenize on
/// `!c.is_alphanumeric() && c != '_'` upstream, so this is a defence-in-depth
/// check rather than the primary filter.
fn word_is_sql_safe(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    !word
        .bytes()
        .any(|b| matches!(b, b'\'' | b'\\' | b'%' | b'_'))
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
        assert!(glob_to_like("' OR 1=1 --").is_none());
        assert!(glob_to_like(r"back\slash").is_none());
        assert!(glob_to_like("raw_underscore").is_none());
        assert!(glob_to_like("raw%percent").is_none());
    }

    #[test]
    fn word_is_sql_safe_accepts_plain_alphanumeric() {
        assert!(word_is_sql_safe("recip"));
        assert!(word_is_sql_safe("rq"));
        assert!(word_is_sql_safe("R3"));
    }

    #[test]
    fn word_is_sql_safe_rejects_sql_specials() {
        assert!(!word_is_sql_safe("' OR 1=1"));
        assert!(!word_is_sql_safe(r"back\slash"));
        assert!(!word_is_sql_safe("with_underscore"));
        assert!(!word_is_sql_safe("with%percent"));
        assert!(!word_is_sql_safe(""));
    }
}

#[derive(Debug, Clone)]
pub struct ChunkHit {
    pub chunk_id: [u8; 16],
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
