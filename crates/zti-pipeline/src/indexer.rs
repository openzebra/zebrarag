use std::path::Path;

use anyhow::Result;
use tracing::info;

use zti_common::ids::project_id;
use zti_dsl::{DslChunker, ProjectIndex, build_index};
use zti_embed::EmbedEngine;
use zti_rerank::TurboReranker;
use zti_store::Db;

use crate::manifest::{FileSnapshot, walk_source_files};
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

    let dsl_index = build_index(&root_str)?;
    info!(
        "dsl-graph: {} symbols, {} edges, {} files",
        dsl_index.symbols.len(),
        dsl_index.edges.len(),
        dsl_index.files.len(),
    );

    let snapshots = walk_source_files(root);
    info!("found {} source files", snapshots.len());

    let chunker = DslChunker::new(&dsl_index);

    let mut all_chunks: Vec<zti_dsl::chunking::Chunk> = Vec::new();
    let mut pending_chunks: Vec<zti_dsl::chunking::Chunk> = Vec::new();

    for (rel, snap) in &snapshots {
        let full_path = root.join(rel);
        let label = full_path.display().to_string();
        let chunks = chunker.chunks_for_file(&label, &snap.contents);
        if !chunks.is_empty() {
            pending_chunks.extend(chunks);
        }
    }

    info!("collected {} chunks from {} files", pending_chunks.len(), snapshots.len());

    reporter.start(pending_chunks.len() as u64);

    let batch_size = 32;
    let mut total_embedded = 0usize;
    let mut reranker = TurboReranker::new(engine.dim())?;

    let mut iter = pending_chunks.into_iter();
    while let Some(first) = iter.next() {
        let mut batch = vec![first];
        while batch.len() < batch_size {
            match iter.next() {
                Some(c) => batch.push(c),
                None => break,
            }
        }

        let bodies: Vec<String> = batch.iter().map(|c| c.embed_text()).collect();
        let bodies_ref: Vec<&str> = bodies.iter().map(|s| s.as_str()).collect();

        match engine.embed_batch(&bodies_ref) {
            Ok(embs) => {
                for (chunk, emb) in batch.into_iter().zip(embs) {
                    if emb.iter().any(|v| v.is_nan()) {
                        tracing::warn!("NaN in embedding for {}:{}-{}, skipping", chunk.file, chunk.start_line, chunk.end_line);
                        reporter.inc(1);
                        continue;
                    }
                    reranker.encode(&emb)?;
                    all_chunks.push(chunk);
                    total_embedded += 1;
                }
            }
            Err(e) => {
                tracing::warn!("embed_batch failed: {}", e);
            }
        }
        reporter.inc(batch_size as u64);
    }

    reporter.finish_with_message(&format!("embedded {} passages", total_embedded));

    let elapsed = start.elapsed();
    info!(
        "indexing complete: {} chunks, {} files in {:.1}s",
        total_embedded,
        snapshots.len(),
        elapsed.as_secs_f64(),
    );

    Ok(IndexStats {
        total_chunks: all_chunks.len(),
        total_files: snapshots.len(),
        new_chunks: total_embedded,
        reindexed_files: 0,
        duration_ms: elapsed.as_millis() as u64,
    })
}
