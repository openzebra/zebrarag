use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{
    BinaryBuilder, FixedSizeBinaryBuilder, Float32Array, ListBuilder, RecordBatch, StringArray,
    StringBuilder, UInt32Array, UInt32Builder, UInt64Array,
};
use tracing::info;

use zti_common::ids::project_id;
use zti_dsl::{DslChunker, EdgeKind, SourceFile, Target, build_index_from_sources};
use zti_embed::EmbedEngine;
use zti_rerank::TurboReranker;
use zti_store::Db;
use zti_tree_sitter::Language;

const APPENDIX_DEPTH: usize = 2;
const APPENDIX_CAP_PER_CHUNK: usize = 32;

use crate::manifest::{FileSnapshot, detect_changes, walk_source_files};
use crate::progress::ProgressReporter;

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

    let need_reindex: Vec<String> = changes
        .added
        .iter()
        .chain(changes.modified.iter())
        .cloned()
        .collect();

    let to_delete: Vec<&str> = changes
        .removed
        .iter()
        .chain(changes.modified.iter())
        .map(|s| s.as_str())
        .collect();

    if !to_delete.is_empty() {
        let chunks_table = db.chunks_table(engine.dim()).await?;
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
    // time inside `zti_dsl::build_index`.
    let dsl_sources = snapshots.iter().map(|(rel, snap)| SourceFile {
        full_path: root.join(rel).display().to_string(),
        content: snap.contents.as_str(),
        language: snap.language,
    });
    let dsl_index = build_index_from_sources(root_str.to_string(), dsl_sources);
    info!(
        "dsl-graph: {} symbols, {} edges, {} files",
        dsl_index.symbols.len(),
        dsl_index.edges.len(),
        dsl_index.files.len(),
    );

    let chunker = DslChunker::new(&dsl_index);

    let mut all_pending: Vec<(zti_dsl::chunking::Chunk, Language)> = Vec::new();
    for rel in &need_reindex {
        let snap = match snapshots.get(rel) {
            Some(s) => s,
            None => continue,
        };
        let full_path = root.join(rel);
        let label = full_path.display().to_string();
        let chunks = chunker.chunks_for_file(&label, &snap.contents);
        for c in chunks {
            all_pending.push((c, snap.language));
        }
    }

    info!("collected {} chunks from {} files", all_pending.len(), need_reindex.len());

    let chunk_sym_set: HashSet<u32> =
        all_pending.iter().map(|(c, _)| c.sym_id).collect();

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

    reporter.start(all_pending.len() as u64);

    let batch_size = 32;
    let mut total_embedded = 0usize;
    let reranker = TurboReranker::new(engine.dim())?;
    let mut chunks_table = db.chunks_table(engine.dim()).await?;

    let mut iter = all_pending.into_iter();
    while let Some((first_chunk, first_lang)) = iter.next() {
        let mut batch_items = vec![(first_chunk, first_lang)];
        while batch_items.len() < batch_size {
            match iter.next() {
                Some((c, l)) => batch_items.push((c, l)),
                None => break,
            }
        }

        let bodies: Vec<String> = batch_items.iter().map(|(c, _)| c.embed_text()).collect();
        let bodies_ref: Vec<&str> = bodies.iter().map(|s| s.as_str()).collect();

        match engine.embed_batch_async(&bodies_ref).await {
            Ok(embs) => {
                let dim = engine.dim();
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;

                let n = batch_items.len();
                let mut chunk_id_builder = FixedSizeBinaryBuilder::new(16);
                let mut file_path_builder = StringBuilder::new();
                let mut language_builder = StringBuilder::new();
                let mut symbol_qualified_builder = StringBuilder::new();
                let mut symbol_kind_builder = StringBuilder::new();
                let mut sym_id_builder = UInt32Array::builder(n);
                let mut parent_sym_id_builder = UInt32Array::builder(n);
                let mut appendix_sym_ids_builder = ListBuilder::new(UInt32Builder::new());
                let mut start_line_builder = UInt32Array::builder(n);
                let mut end_line_builder = UInt32Array::builder(n);
                let mut content_builder = StringBuilder::new();
                let mut turbo_code_builder = BinaryBuilder::new();
                let mut indexed_at_builder = UInt64Array::builder(n);
                let mut embeddings: Vec<f32> = Vec::with_capacity(n * dim);

                let zipped: Vec<_> = batch_items.into_iter().zip(embs).collect();

                for ((chunk, lang), emb) in zipped {
                    if emb.iter().any(|v| v.is_nan()) {
                        tracing::warn!(
                            "NaN in embedding for {}:{}-{}, skipping",
                            chunk.file,
                            chunk.start_line,
                            chunk.end_line
                        );
                        reporter.inc(1);
                        continue;
                    }

                    let mut hasher = blake3::Hasher::new();
                    hasher.update(chunk.file.as_bytes());
                    hasher.update(&chunk.start_line.to_le_bytes());
                    hasher.update(&chunk.end_line.to_le_bytes());
                    hasher.update(chunk.qualified.as_bytes());
                    let hash = hasher.finalize();
                    let mut chunk_id = [0u8; 16];
                    chunk_id.copy_from_slice(&hash.as_bytes()[..16]);

                    let turbo = match reranker.encode(&emb) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            tracing::debug!("turbo encode failed: {}", e);
                            None
                        }
                    };

                    let parent_sym_id = dsl_index
                        .symbols
                        .get(chunk.sym_id as usize)
                        .and_then(|s| s.parent);
                    let appendix_ids = appendix_for(chunk.sym_id);

                    chunk_id_builder.append_value(chunk_id)?;
                    file_path_builder.append_value(&chunk.file);
                    language_builder.append_value(lang.as_str());
                    symbol_qualified_builder.append_value(&chunk.qualified);
                    symbol_kind_builder.append_value(chunk.kind.as_str());
                    sym_id_builder.append_value(chunk.sym_id);
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
                    embeddings.extend_from_slice(&emb);

                    total_embedded += 1;
                }

                if !embeddings.is_empty() {
                    let embedding_arr = arrow::array::FixedSizeListArray::new(
                        std::sync::Arc::new(arrow::datatypes::Field::new(
                            "item",
                            arrow::datatypes::DataType::Float32,
                            false,
                        )),
                        dim as i32,
                        std::sync::Arc::new(Float32Array::from(embeddings)),
                        None,
                    );

                    let record = RecordBatch::try_new(
                        std::sync::Arc::new(zti_store::schema::chunks_schema(dim)),
                        vec![
                            std::sync::Arc::new(chunk_id_builder.finish()),
                            std::sync::Arc::new(file_path_builder.finish()),
                            std::sync::Arc::new(language_builder.finish()),
                            std::sync::Arc::new(symbol_qualified_builder.finish()),
                            std::sync::Arc::new(symbol_kind_builder.finish()),
                            std::sync::Arc::new(sym_id_builder.finish()),
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
                tracing::warn!("embed_batch failed: {}", e);
            }
        }
        reporter.inc(batch_size as u64);
    }

    chunks_table.index_vector().await?;

    upsert_files(&files_table, &snapshots, &need_reindex).await?;

    let languages: HashSet<Language> = snapshots.values().map(|s| s.language).collect();
    let languages: Vec<&Language> = languages.iter().collect();
    upsert_project(db, &pid, root_str, total_embedded, snapshots.len(), &languages, engine).await?;

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
                std::sync::Arc::new(StringArray::from(vec![snap.language.as_str()])),
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

async fn upsert_project(
    db: &Db,
    pid: &[u8; 32],
    root: std::borrow::Cow<'_, str>,
    total_chunks: usize,
    total_files: usize,
    languages: &[&Language],
    engine: &EmbedEngine,
) -> Result<()> {
    let projects_table = db.projects_table().await?;
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let mut project_id_builder = FixedSizeBinaryBuilder::new(32);
    project_id_builder.append_value(pid)?;

    let lang_values = StringArray::from(
        languages.iter().map(|l| l.as_str()).collect::<Vec<_>>(),
    );
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
        ],
    )?;

    projects_table.upsert(record).await?;
    Ok(())
}
