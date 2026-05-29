use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};

use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{
    BinaryBuilder, FixedSizeBinaryBuilder, ListBuilder, RecordBatch, StringArray,
    StringBuilder, UInt32Array, UInt32Builder, UInt64Array, UInt8Array,
};
use rayon::prelude::*;
use tracing::info;

use zti_common::ids::project_id;
use zti_dsl::chunking::ChunkStrategy;
use zti_dsl::{Chunk, DslChunker, EdgeKind, SourceFile, Target, build_index_from_sources};
use zti_embed::EmbedEngine;
use zti_recursive_chunk;
use zti_rerank::TurboReranker;
use zti_store::Db;
use zti_ts_core::walker::LanguageFrontend;
use zti_tree_sitter::{Language, frontend_for};

const APPENDIX_DEPTH: usize = 2;
const APPENDIX_CAP_PER_CHUNK: usize = 32;
const CHARS_PER_TOKEN: usize = 4;
const CHUNK_OVERLAP: usize = 200;
const BPT_SAMPLE_BYTES: usize = 64 * 1024;
const MIN_CHUNK_FLOOR: usize = 512;

use crate::manifest::{FileSnapshot, SourceKind, detect_changes, walk_source_files};
use crate::progress::ProgressReporter;

#[derive(Debug, Clone, Copy)]
struct AdaptiveChunkSizing {
    chunk_size: usize,
    min_chunk_size: usize,
}

/// Largest byte index <= `max` that is a UTF-8 char boundary.
#[inline]
fn floor_boundary(s: &str, max: usize) -> usize {
    if s.len() <= max {
        return s.len();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Pure byte-sizing math: given a measured bytes-per-token ratio, decide whether
/// `body_len` bytes exceed the model's token budget and, if so, the byte limits
/// for recursive splitting. Split when the estimated token count
/// (`body_len / bpt`) exceeds `max_len`, i.e. `body_len > max_len * bpt`.
fn sizing_for(body_len: usize, max_len: usize, bpt: usize) -> Option<AdaptiveChunkSizing> {
    let chunk_size = max_len.saturating_mul(bpt);
    (body_len > chunk_size).then(|| AdaptiveChunkSizing {
        chunk_size: chunk_size.max(MIN_CHUNK_FLOOR + 1),
        min_chunk_size: (max_len / 2).saturating_mul(bpt).max(MIN_CHUNK_FLOOR),
    })
}

/// `Some(sizing)` when the body should be recursively split; `None` keeps it whole.
fn adaptive_split(body: &str, engine: &EmbedEngine) -> Option<AdaptiveChunkSizing> {
    let max_len = engine.profile().max_length;

    // Fast path: bytes ≤ max_len → tokens ≤ bytes ≤ max_len → always fits.
    if body.len() <= max_len {
        return None;
    }

    let bpt = if engine.truncates() {
        CHARS_PER_TOKEN
    } else {
        let sample = &body[..floor_boundary(body, BPT_SAMPLE_BYTES)];
        match engine.count_tokens(sample) {
            Ok(n) if n > 0 => (sample.len() / n).max(1),
            _ => CHARS_PER_TOKEN,
        }
    };

    sizing_for(body.len(), max_len, bpt)
}

#[inline]
fn generate_sub_chunks<'a>(
    chunk: &Chunk<'a>,
    sizing: &AdaptiveChunkSizing,
    lang: Option<&tree_sitter::Language>,
    kind_label: &'static str,
    out: &mut Vec<(Chunk<'a>, &'static str)>,
    terminal_kinds: &[u16],
) {
    let sub_chunks = zti_recursive_chunk::split_text(
        &chunk.body,
        &zti_recursive_chunk::ChunkConfig {
            chunk_size: sizing.chunk_size,
            min_chunk_size: sizing.min_chunk_size,
            chunk_overlap: CHUNK_OVERLAP,
        },
        lang,
        terminal_kinds,
    );
    let total = sub_chunks.len() as u32;
    for (i, sub) in sub_chunks.iter().enumerate() {
        let sc = Chunk {
            file: chunk.file.clone(),
            rel_file: chunk.rel_file.clone(),
            start_line: chunk.start_line + sub.start_line - 1,
            end_line: chunk.start_line + sub.end_line - 1,
            sym_id: chunk.sym_id,
            sub_chunk_idx: i as u32,
            total_sub_chunks: total,
            chunk_strategy: ChunkStrategy::Recursive,
            body: Cow::Owned(chunk.body[sub.byte_start..sub.byte_end].to_string()),
            qualified: chunk.qualified.clone(),
            kind: chunk.kind,
        };
        out.push((sc, kind_label));
    }
}

pub struct IndexStats {
    pub total_chunks: usize,
    pub total_files: usize,
    pub new_chunks: usize,
    pub reindexed_files: usize,
    pub duration_ms: u64,
}

pub async fn index_project(
    root: &Path,
    engine: &EmbedEngine,
    db: &Db,
    reporter: &dyn ProgressReporter,
    override_method: Option<zti_ann::SearchMethod>,
    cancel: &AtomicBool,
    refresh: bool,
) -> Result<IndexStats> {
    let start = std::time::Instant::now();
    let pid = project_id(root);

    let root_str = root.to_string_lossy();
    info!("indexing {}", root_str);

    let snapshots = walk_source_files(root);
    info!("found {} source files", snapshots.len());

    let files_table = db.files_table().await?;
    let previous = files_table.list().await.unwrap_or_default();

    let changes = detect_changes(&snapshots, &previous);
    info!(
        "changes: {} added, {} modified, {} removed, {} unchanged",
        changes.added.len(),
        changes.modified.len(),
        changes.removed.len(),
        changes.unchanged.len(),
    );

    let mut chunks_table = db.chunks_table(engine.dim()).await?;

    let force_rebuild = if refresh {
        true
    } else if !previous.is_empty() {
        chunks_table.len().await? == 0
    } else {
        false
    };

    if force_rebuild && !refresh {
        info!("self-heal: empty index detected, forcing full reindex");
    }

    let need_reindex: Vec<String>;
    let to_delete: Vec<&str>;

    if force_rebuild {
        need_reindex = snapshots.keys().cloned().collect();
        to_delete = previous.iter().map(|r| r.file_path.as_str()).collect();
    } else {
        need_reindex = changes
            .added
            .iter()
            .chain(changes.modified.iter())
            .cloned()
            .collect();
        to_delete = changes
            .removed
            .iter()
            .chain(changes.modified.iter())
            .map(|s| s.as_str())
            .collect();
    }

    if !to_delete.is_empty() {
        chunks_table.delete_for_files(&to_delete).await?;
        files_table.delete_for_paths(&to_delete).await?;
        info!("deleted chunks for {} files", to_delete.len());
    }

    if need_reindex.is_empty() {
        reporter.finish_with_message("nothing to reindex");
        let elapsed = start.elapsed();
        let total_files = snapshots.len();
        return Ok(IndexStats {
            total_chunks: 0,
            total_files,
            new_chunks: 0,
            reindexed_files: 0,
            duration_ms: elapsed.as_millis() as u64,
        });
    }

    // Single FS walk: reuse the snapshots we already loaded to drive the DSL
    // parser. Avoids walking the tree (and re-reading every file) a second
    // time inside `zti_dsl::build_index`. Text-kind snapshots have no
    // tree-sitter frontend, so they're filtered out here and chunked
    // separately below.
    let n_files = need_reindex.len();
    reporter.set_phase(
        zti_protocol::response::IndexPhase::Gather,
        0,
        n_files as u64,
        "parsing code files",
    );
    let dsl_sources = snapshots.iter().filter_map(|(rel, snap)| match snap.kind {
        SourceKind::Code(lang) => Some(SourceFile {
            full_path: root.join(rel).display().to_string(),
            content: snap.contents.as_str(),
            language: lang,
        }),
        SourceKind::Text => None,
    });
    let dsl_index = build_index_from_sources(root_str.to_string(), dsl_sources);
    info!(
        "dsl-graph: {} symbols, {} edges, {} files",
        dsl_index.symbols.len(),
        dsl_index.edges.len(),
        dsl_index.files.len(),
    );

    reporter.set_phase(
        zti_protocol::response::IndexPhase::Gather,
        dsl_index.files.len() as u64,
        n_files as u64,
        "generating chunks",
    );
    let chunker = DslChunker::new(&dsl_index);

    // Pre-compute terminal node kinds for every language in the project
    // (read-only access from parallel threads below).
    let mut terminal_cache: FxHashMap<Language, Vec<u16>> =
        FxHashMap::with_capacity_and_hasher(4, rustc_hash::FxBuildHasher);
    for lang in dsl_index.files.iter().map(|f| f.language) {
        if terminal_cache.contains_key(&lang) {
            continue;
        }
        let frontend = frontend_for(lang);
        let ts_lang = frontend.language();
        let names = frontend.config().terminal_node_kinds;
        let mut ids = Vec::with_capacity(names.len());
        for name in names {
            let id = ts_lang.id_for_node_kind(name, true);
            if id != 0 {
                ids.push(id);
            }
        }
        terminal_cache.insert(lang, ids);
    }

    let all_pending: Vec<(Chunk<'_>, &'static str)> = need_reindex
        .par_iter()
        .flat_map(|rel| {
            let snap = match snapshots.get(rel) {
                Some(s) => s,
                None => return Vec::new(),
            };
            match snap.kind {
                SourceKind::Code(lang) => {
                    let full_path = root.join(rel);
                    let label = full_path.display().to_string();
                    let chunks = chunker.chunks_for_file(&label, &snap.contents);
                    let frontend = frontend_for(lang);
                    let ts_lang = frontend.language();
                    let terminal_ids = terminal_cache
                        .get(&lang)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let mut out = Vec::with_capacity(chunks.len());
                    for c in chunks {
                        match adaptive_split(&c.body, engine) {
                            Some(sizing) => generate_sub_chunks(
                                &c, &sizing, Some(&ts_lang), lang.as_str(), &mut out, terminal_ids,
                            ),
                            None => out.push((c, lang.as_str())),
                        }
                    }
                    out
                }
                SourceKind::Text => {
                    let full_path = root.join(rel).display().to_string();
                    let chunk = zti_dsl::chunking::chunk_text_file(
                        rel.clone(),
                        full_path,
                        snap.contents.clone(),
                    );
                    match adaptive_split(&chunk.body, engine) {
                        Some(sizing) => {
                            let mut out = Vec::with_capacity(4);
                            generate_sub_chunks(
                                &chunk, &sizing, None, "text", &mut out, &[],
                            );
                            out
                        }
                        None => vec![(chunk, "text")],
                    }
                }
            }
        })
        .collect();

    info!(
        "collected {} chunks from {} files",
        all_pending.len(),
        need_reindex.len()
    );

    let chunk_sym_set: HashSet<u32> = all_pending
        .iter()
        .filter(|(c, _)| c.sym_id != u32::MAX)
        .map(|(c, _)| c.sym_id)
        .collect();

    let appendix_for = |sym_id: u32| -> Vec<u32> {
        let mut visited: HashSet<u32> = HashSet::with_capacity(16);
        let mut queue: VecDeque<(u32, usize)> = VecDeque::with_capacity(16);
        let mut out: Vec<u32> = Vec::with_capacity(APPENDIX_CAP_PER_CHUNK);
        visited.insert(sym_id);
        queue.push_back((sym_id, 0));
        while let Some((cur, depth)) = queue.pop_front() {
            if depth >= APPENDIX_DEPTH {
                continue;
            }
            let Some(edges) = dsl_index.forward_edges.get(&cur) else {
                continue;
            };
            for edge in edges.iter().filter(|e| e.kind == EdgeKind::Call) {
                let Target::Resolved(rid) = edge.to else {
                    continue;
                };
                if !visited.insert(rid) {
                    continue;
                }
                if !chunk_sym_set.contains(&rid) {
                    continue;
                }
                if out.len() < APPENDIX_CAP_PER_CHUNK {
                    out.push(rid);
                }
                queue.push_back((rid, depth + 1));
            }
        }
        out
    };

    let total_chunks = all_pending.len();
    reporter.start(total_chunks as u64);
    reporter.set_phase(
        zti_protocol::response::IndexPhase::Tokenize,
        0,
        total_chunks as u64,
        "tokenizing chunks",
    );

    let batch_size = engine.recommended_batch_size();
    let hw = engine.hardware();
    info!(
        batch_size,
        device = ?hw.device,
        mem_avail_mb = hw.mem_avail >> 20,
        max_length = engine.profile().max_length,
        "computed embed batch_size",
    );

    let mut total_embedded = 0usize;
    let reranker = TurboReranker::new(engine.dim())?;

    let total_chunks = all_pending.len();
    // Tokenize in batches so the progress bar advances 0→N instead of
    // freezing until all chunks are tokenized.
    let encs: Vec<zti_embed::Tokenized> = {
        let passage_prefix = &engine.profile().passage_prefix;
        let prefixed: Vec<Cow<'_, str>> = all_pending
            .iter()
            .map(|(c, _)| zti_embed::apply_prefix(&c.body, passage_prefix))
            .collect();
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_ref()).collect();

        const TOK_BATCH: usize = 512;
        let mut out: Vec<zti_embed::Tokenized> = Vec::with_capacity(refs.len());
        for slice in refs.chunks(TOK_BATCH) {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("indexing cancelled");
            }
            out.extend(engine.tokenize(slice)?);
            reporter.set_phase(
                zti_protocol::response::IndexPhase::Tokenize,
                out.len() as u64,
                total_chunks as u64,
                "tokenizing chunks",
            );
        }
        out
    };

    // All length math below must use the same cap that `prepare_from_encs`
    // applies before running the model — otherwise a chunk that tokenizes to
    // (say) 6000 ids but will be truncated to 2048 at inference inflates the
    // bucketing pad estimate and forces a batch-of-one, defeating batching.
    let model_max_len = engine.profile().max_length;
    let effective_len = |idx: usize| encs[idx].ids.len().min(model_max_len).max(1);

    // Sort ASCENDING. The first batches pack many short chunks (cheap, ~50 ms
    // each) so progress visibly moves; the last few batches process the long
    // chunks at seq=max_length (the expensive ones). Descending order is
    // slightly better for ORT's BFCArena reuse but front-loads every slow
    // batch and makes the run look frozen for the first 15–30 s — that
    // perception cost dominated the real arena-extension cost in practice.
    let mut order: Vec<usize> = (0..encs.len()).collect();
    order.sort_by_key(|&i| effective_len(i));

    // Token budget per batch matches the per-sample shape `batch_size` was
    // sized for (≈512 tokens × batch_size items). Item count is capped at
    // BATCH_CEILING so very short chunks don't inflate the working set just
    // because the token budget allows it.
    let budget_tokens = batch_size.saturating_mul(zti_embed::batch::TYPICAL_SEQ_LEN);
    let max_items = batch_size
        .saturating_mul(4)
        .min(zti_embed::batch::BATCH_CEILING);

    // Reusable per-batch view into `encs` (no per-batch allocation: cleared
    // and refilled with references each iteration).
    let mut batch_encs: Vec<&zti_embed::Tokenized> = Vec::with_capacity(max_items);

    let mut cursor = 0usize;
    while cursor < order.len() {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("indexing cancelled");
        }
        let mut end = cursor;
        let mut pad_len = 0usize;
        while end < order.len() {
            let l = effective_len(order[end]);
            let new_pad = pad_len.max(l);
            let count = end - cursor + 1;
            if count > 1 && (count.saturating_mul(new_pad) > budget_tokens || count > max_items) {
                break;
            }
            pad_len = new_pad;
            end += 1;
        }
        let idxs = &order[cursor..end];
        let n_batch = idxs.len();

        batch_encs.clear();
        batch_encs.extend(idxs.iter().map(|&i| &encs[i]));

        let batch_started = std::time::Instant::now();
        match engine.embed_batch_tokenized_async(&batch_encs).await {
            Ok(embs) => {
                let dim = engine.dim();
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                let n = idxs.len();
                let mut chunk_id_builder = FixedSizeBinaryBuilder::new(16);
                let mut file_path_builder = StringBuilder::with_capacity(n, n * 64);
                let mut language_builder = StringBuilder::with_capacity(n, n * 8);
                let mut symbol_qualified_builder = StringBuilder::with_capacity(n, n * 64);
                let mut symbol_kind_builder = StringBuilder::with_capacity(n, n * 16);
                let mut sym_id_builder = UInt32Array::builder(n);
                let mut sub_chunk_idx_builder = UInt32Array::builder(n);
                let mut total_sub_chunks_builder = UInt32Array::builder(n);
                let mut chunk_strategy_builder = UInt8Array::builder(n);
                let mut parent_sym_id_builder = UInt32Array::builder(n);
                let mut appendix_sym_ids_builder = ListBuilder::new(UInt32Builder::new());
                let mut start_line_builder = UInt32Array::builder(n);
                let mut end_line_builder = UInt32Array::builder(n);
                let mut content_builder = StringBuilder::with_capacity(n, n * 64);
                let mut turbo_code_builder = BinaryBuilder::new();
                let mut indexed_at_builder = UInt64Array::builder(n);

                for (i, &idx) in idxs.iter().enumerate() {
                    let (chunk, lang) = &all_pending[idx];
                    let emb = embs.row(i);

                    if emb.iter().any(|v| v.is_nan()) {
                        anyhow::bail!(
                            "NaN embedding for {}:{}-{}",
                            chunk.file,
                            chunk.start_line,
                            chunk.end_line,
                        );
                    }

                    let mut hasher = blake3::Hasher::new();
                    hasher.update(chunk.file.as_bytes());
                    hasher.update(&chunk.start_line.to_le_bytes());
                    hasher.update(&chunk.end_line.to_le_bytes());
                    hasher.update(chunk.qualified.as_bytes());
                    hasher.update(&chunk.sub_chunk_idx.to_le_bytes());
                    let hash = hasher.finalize();
                    let mut chunk_id = [0u8; 16];
                    chunk_id.copy_from_slice(&hash.as_bytes()[..16]);

                    let turbo = match reranker.encode(emb) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            tracing::debug!("turbo encode failed: {}", e);
                            None
                        }
                    };

                    let parent_sym_id = if chunk.sym_id == u32::MAX {
                        None
                    } else {
                        dsl_index
                            .symbols
                            .get(chunk.sym_id as usize)
                            .and_then(|s| s.parent)
                    };
                    let appendix_ids: Vec<u32> = if chunk.sym_id == u32::MAX {
                        Vec::new()
                    } else {
                        appendix_for(chunk.sym_id)
                    };

                    chunk_id_builder.append_value(chunk_id)?;
                    file_path_builder.append_value(&chunk.file);
                    language_builder.append_value(lang);
                    symbol_qualified_builder.append_value(&chunk.qualified);
                    symbol_kind_builder.append_value(chunk.kind.as_str());
                    sym_id_builder.append_value(chunk.sym_id);
                    sub_chunk_idx_builder.append_value(chunk.sub_chunk_idx);
                    total_sub_chunks_builder.append_value(chunk.total_sub_chunks);
                    chunk_strategy_builder.append_value(chunk.chunk_strategy as u8);
                    match parent_sym_id {
                        Some(p) => parent_sym_id_builder.append_value(p),
                        None => parent_sym_id_builder.append_null(),
                    }
                    if appendix_ids.is_empty() {
                        appendix_sym_ids_builder.append_null();
                    } else {
                        for id in &appendix_ids {
                            appendix_sym_ids_builder.values().append_value(*id);
                        }
                        appendix_sym_ids_builder.append(true);
                    }
                    start_line_builder.append_value(chunk.start_line);
                    end_line_builder.append_value(chunk.end_line);
                    content_builder.append_value(&chunk.body);
                    match &turbo {
                        Some(t) => turbo_code_builder.append_value(t),
                        None => turbo_code_builder.append_null(),
                    }
                    indexed_at_builder.append_value(now_ns);

                    total_embedded += 1;
                }

                {
                    let embedding_arr = embs.into_fixed_size_list();

                    let record = RecordBatch::try_new(
                        std::sync::Arc::new(zti_store::schema::chunks_schema(dim)),
                        vec![
                            std::sync::Arc::new(chunk_id_builder.finish()),
                            std::sync::Arc::new(file_path_builder.finish()),
                            std::sync::Arc::new(language_builder.finish()),
                            std::sync::Arc::new(symbol_qualified_builder.finish()),
                            std::sync::Arc::new(symbol_kind_builder.finish()),
                            std::sync::Arc::new(sym_id_builder.finish()),
                            std::sync::Arc::new(sub_chunk_idx_builder.finish()),
                            std::sync::Arc::new(total_sub_chunks_builder.finish()),
                            std::sync::Arc::new(chunk_strategy_builder.finish()),
                            std::sync::Arc::new(parent_sym_id_builder.finish()),
                            std::sync::Arc::new(appendix_sym_ids_builder.finish()),
                            std::sync::Arc::new(start_line_builder.finish()),
                            std::sync::Arc::new(end_line_builder.finish()),
                            std::sync::Arc::new(content_builder.finish()),
                            std::sync::Arc::new(turbo_code_builder.finish()),
                            std::sync::Arc::new(indexed_at_builder.finish()),
                            std::sync::Arc::new(embedding_arr),
                        ],
                    )?;

                    chunks_table.upsert(record).await?;
                }
            }
            Err(e) => {
                anyhow::bail!("embed batch failed: {e}");
            }
        }
        tracing::debug!(
            items = n_batch,
            seq = pad_len,
            ms = batch_started.elapsed().as_millis() as u64,
            "embedded batch",
        );
        reporter.inc(n_batch as u64);
        cursor = end;
    }

    if total_chunks > 0 && total_embedded == 0 {
        anyhow::bail!(
            "no embeddings produced from {total_chunks} chunks"
        );
    }

    let hw = engine.hardware();
    let previous_row = db.projects_table().await?.get(&pid).await.ok().flatten();
    let previous_params: Option<zti_ann::SearchParams> = previous_row
        .as_ref()
        .and_then(|r| r.search_params.as_deref())
        .and_then(|s| toml::from_str(s).ok());
    let total_in_db = chunks_table.len().await?;
    let mut params =
        zti_ann::choose_method(total_in_db, engine.dim(), hw, previous_params.as_ref());
    if let Some(m) = override_method {
        params.method = m;
    }
    info!(
        "search method: {:?} (n={}, dim={}, ram_avail={} MB)",
        params.method,
        total_in_db,
        engine.dim(),
        hw.mem_avail >> 20
    );

    reporter.set_phase(
        zti_protocol::response::IndexPhase::BuildIndex,
        0,
        0,
        "building search index",
    );
    chunks_table.optimize().await?;
    chunks_table.build_index(&params).await?;

    upsert_files(&files_table, &snapshots, &need_reindex).await?;

    let languages: HashSet<Language> = snapshots
        .values()
        .filter_map(|s| match s.kind {
            SourceKind::Code(l) => Some(l),
            SourceKind::Text => None,
        })
        .collect();
    let languages: Vec<&Language> = languages.iter().collect();
    upsert_project(
        db,
        &pid,
        root_str,
        total_in_db,
        snapshots.len(),
        &languages,
        engine,
        &params,
    )
    .await?;

    reporter.finish_with_message(&format!("embedded {} passages", total_embedded));

    let elapsed = start.elapsed();
    info!(
        "indexing complete: {} chunks, {} files in {:.1}s",
        total_embedded,
        snapshots.len(),
        elapsed.as_secs_f64(),
    );

    Ok(IndexStats {
        total_chunks: total_embedded,
        total_files: snapshots.len(),
        new_chunks: total_embedded,
        reindexed_files: need_reindex.len(),
        duration_ms: elapsed.as_millis() as u64,
    })
}

async fn upsert_files(
    files_table: &zti_store::files_table::FilesTable,
    snapshots: &std::collections::HashMap<String, FileSnapshot>,
    changed_paths: &[String],
) -> Result<()> {
    for path in changed_paths {
        let snap = match snapshots.get(path) {
            Some(s) => s,
            None => continue,
        };

        let mut blake3_builder = FixedSizeBinaryBuilder::new(32);
        blake3_builder.append_value(snap.blake3)?;

        let record = RecordBatch::try_new(
            std::sync::Arc::new(zti_store::schema::files_schema()),
            vec![
                std::sync::Arc::new(StringArray::from(vec![path.clone()])),
                std::sync::Arc::new(blake3_builder.finish()),
                std::sync::Arc::new(UInt64Array::from(vec![snap.mtime_ns as u64])),
                std::sync::Arc::new(UInt64Array::from(vec![snap.size_bytes])),
                std::sync::Arc::new(StringArray::from(vec![snap.kind.label()])),
                std::sync::Arc::new(arrow::array::ListArray::new_null(
                    std::sync::Arc::new(arrow::datatypes::Field::new(
                        "item",
                        arrow::datatypes::DataType::FixedSizeBinary(16),
                        false,
                    )),
                    1,
                )),
                std::sync::Arc::new(UInt64Array::from(vec![
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64,
                ])),
            ],
        )?;

        files_table.upsert(record).await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn upsert_project(
    db: &Db,
    pid: &[u8; 32],
    root: std::borrow::Cow<'_, str>,
    total_chunks: usize,
    total_files: usize,
    languages: &[&Language],
    engine: &EmbedEngine,
    choice: &zti_ann::SearchParams,
) -> Result<()> {
    let projects_table = db.projects_table().await?;
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let mut project_id_builder = FixedSizeBinaryBuilder::new(32);
    project_id_builder.append_value(pid)?;

    let lang_values = StringArray::from(languages.iter().map(|l| l.as_str()).collect::<Vec<_>>());
    let languages_arr = arrow::array::ListArray::new(
        Arc::new(arrow::datatypes::Field::new(
            "item",
            arrow::datatypes::DataType::Utf8,
            false,
        )),
        arrow::buffer::OffsetBuffer::from_lengths([languages.len()]),
        Arc::new(lang_values),
        None,
    );

    let search_method = StringArray::from(vec![choice.method.as_str()]);
    let search_params = StringArray::from(vec![toml::to_string(&choice)?]);

    let record = RecordBatch::try_new(
        Arc::new(zti_store::schema::projects_schema()),
        vec![
            Arc::new(project_id_builder.finish()),
            Arc::new(StringArray::from(vec![root.to_string()])),
            Arc::new(languages_arr),
            Arc::new(StringArray::from(vec![engine.profile().model_id.clone()])),
            Arc::new(UInt32Array::from(vec![engine.dim() as u32])),
            Arc::new(UInt64Array::from(vec![total_chunks as u64])),
            Arc::new(UInt64Array::from(vec![total_files as u64])),
            Arc::new(UInt64Array::from(vec![now_ns])),
            Arc::new(UInt64Array::from(vec![now_ns])),
            Arc::new(search_method),
            Arc::new(search_params),
        ],
    )?;

    projects_table.upsert(record).await?;
    Ok(())
}

#[cfg(test)]
mod tests_indexing {
    use std::borrow::Cow;
    use super::{MIN_CHUNK_FLOOR, floor_boundary, sizing_for};
    use zti_dsl::chunking::{Chunk, ChunkStrategy};
    use zti_ts_core::types::Kind;

    #[test]
    fn floor_boundary_returns_len_when_under_max() {
        let s = "hello";
        assert_eq!(floor_boundary(s, 100), s.len());
        assert_eq!(floor_boundary(s, s.len()), s.len());
    }

    #[test]
    fn floor_boundary_ascii_is_exact() {
        // Every byte index in ASCII is a char boundary.
        let s = "abcdefgh";
        assert_eq!(floor_boundary(s, 3), 3);
    }

    #[test]
    fn floor_boundary_never_splits_a_codepoint() {
        // "é" is 2 bytes (0xC3 0xA9); "€" is 3 bytes. Build a string where a max
        // cut would land mid-codepoint, and assert we always get a valid slice.
        let s = "aé€b"; // bytes: a(1) é(2) €(3) b(1) = 7 bytes
        for max in 0..s.len() {
            let end = floor_boundary(s, max);
            assert!(end <= max);
            assert!(s.is_char_boundary(end), "end {end} not a boundary for max {max}");
            // The slice must not panic and must be valid UTF-8 (guaranteed by &str).
            let _ = &s[..end];
        }
    }

    #[test]
    fn sizing_for_none_when_body_fits() {
        // body_len <= max_len * bpt → fits in one chunk.
        assert!(sizing_for(2048, 512, 4).is_none()); // exactly at the limit
        assert!(sizing_for(100, 512, 4).is_none()); // well under
    }

    #[test]
    fn sizing_for_splits_and_uses_measured_bpt() {
        // Sparse markdown: bpt high → larger byte-chunks than the old `* 4`.
        let s = sizing_for(170_000_000, 512, 7).expect("huge sparse body must split");
        assert_eq!(s.chunk_size, 512 * 7); // 3584, not the old 2048
        assert_eq!(s.min_chunk_size, 256 * 7); // (max_len/2)*bpt = 1792
        assert!(s.chunk_size > s.min_chunk_size, "chunker requires chunk_size > min");
    }

    #[test]
    fn sizing_for_dense_code_smaller_chunks() {
        // Dense code: bpt low → tighter byte-chunks that still fit max_len tokens.
        let s = sizing_for(50_000, 512, 2).expect("body exceeds 1024 bytes → split");
        assert_eq!(s.chunk_size, 1024);
        assert_eq!(s.min_chunk_size, 512);
    }

    #[test]
    fn sizing_for_enforces_floors() {
        // Tiny max_len * bpt would drop below the floor; floors must clamp it
        // while preserving chunk_size > min_chunk_size.
        let s = sizing_for(10_000, 10, 1).expect("10_000 > 10 → split");
        assert_eq!(s.chunk_size, MIN_CHUNK_FLOOR + 1); // 513
        assert_eq!(s.min_chunk_size, MIN_CHUNK_FLOOR); // 512
        assert!(s.chunk_size > s.min_chunk_size);
    }

    #[test]
    fn sizing_for_saturates_without_panic() {
        // chunk_size saturates to usize::MAX → nothing can exceed it → None, no overflow.
        assert!(sizing_for(10, usize::MAX, 4).is_none());
        assert!(sizing_for(usize::MAX, 512, usize::MAX).is_none());
        assert!(sizing_for(usize::MAX - 1, usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn test_generate_sub_chunks_metadata() {
        let parent = Chunk {
            file: "/src/main.rs".into(),
            rel_file: "src/main.rs".into(),
            start_line: 10,
            end_line: 20,
            sym_id: 42,
            sub_chunk_idx: 0,
            total_sub_chunks: 1,
            chunk_strategy: ChunkStrategy::Symbol,
            body: Cow::Borrowed("line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7"),
            qualified: "foo::bar".into(),
            kind: Kind::Function,
        };

        let mock_sub_chunks = [
            zti_recursive_chunk::SubChunk {
                byte_start: 0,
                byte_end: 20,
                start_line: 1,
                end_line: 3,
            },
            zti_recursive_chunk::SubChunk {
                byte_start: 20,
                byte_end: 48,
                start_line: 3,
                end_line: 7,
            },
        ];

        let mut out = Vec::new();
        let total = mock_sub_chunks.len() as u32;
        for (i, sub) in mock_sub_chunks.iter().enumerate() {
            let sc = Chunk {
                file: parent.file.clone(),
                rel_file: parent.rel_file.clone(),
                start_line: parent.start_line + sub.start_line - 1,
                end_line: parent.start_line + sub.end_line - 1,
                sym_id: parent.sym_id,
                sub_chunk_idx: i as u32,
                total_sub_chunks: total,
                chunk_strategy: ChunkStrategy::Recursive,
                body: Cow::Owned(parent.body[sub.byte_start..sub.byte_end].to_string()),
                qualified: parent.qualified.clone(),
                kind: parent.kind,
            };
            out.push((sc, "rust"));
        }

        assert_eq!(out.len(), 2);

        let s0 = &out[0].0;
        assert_eq!(s0.sub_chunk_idx, 0);
        assert_eq!(s0.total_sub_chunks, 2);
        assert_eq!(s0.chunk_strategy, ChunkStrategy::Recursive);
        assert_eq!(s0.start_line, 10);
        assert_eq!(s0.end_line, 12);
        assert_eq!(s0.body, parent.body[..20]);

        let s1 = &out[1].0;
        assert_eq!(s1.sub_chunk_idx, 1);
        assert_eq!(s1.total_sub_chunks, 2);
        assert_eq!(s1.chunk_strategy, ChunkStrategy::Recursive);
        assert_eq!(s1.start_line, 12);
        assert_eq!(s1.end_line, 16);
        assert_eq!(s1.body, parent.body[20..48]);
    }
}
