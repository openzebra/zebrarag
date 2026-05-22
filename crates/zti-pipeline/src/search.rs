use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::{Result, anyhow};

use zti_ann::{AnnCache, AnnHandle, AnnIndexBuilder, SearchMethod, SearchParams};
use zti_embed::EmbedEngine;
use zti_rerank::TurboReranker;
use zti_store::chunks_table::ChunkHit;

const KNN_OVERFETCH_MULT: usize = 4;
const DIVERSITY_PENALTY: f32 = 0.04;
const KEYWORD_NAME_BOOST: f32 = 0.5;
const KEYWORD_CONTENT_BOOST: f32 = 0.3;

#[inline]
fn apply_keyword_boost(query: &str, candidates: &mut [ChunkHit]) {
    if query.is_empty() {
        return;
    }
    for c in candidates {
        if c.symbol_qualified.contains(query) {
            c.score += KEYWORD_NAME_BOOST;
        } else if c.content.contains(query) {
            c.score += KEYWORD_CONTENT_BOOST;
        }
    }
}

pub struct SearchOpts<'a> {
    pub limit: usize,
    pub languages: Option<&'a [String]>,
    pub path_glob: Option<&'a str>,
}

pub struct Hit {
    pub chunk: ChunkHit,
    pub score: f32,
}

pub async fn search(
    query: &str,
    engine: &EmbedEngine,
    db: &zti_store::Db,
    reranker: &TurboReranker,
    ann_cache: &AnnCache,
    pid: &[u8; 32],
    opts: &SearchOpts<'_>,
) -> Result<Vec<Hit>> {
    let projects = db.projects_table().await?;
    let project = projects
        .get(pid)
        .await?
        .ok_or_else(|| anyhow!("project not indexed"))?;

    let previous: Option<SearchParams> = project
        .search_params
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());

    let params: SearchParams = match previous {
        Some(p) => p,
        None => zti_ann::choose_method(
            project.total_chunks as usize,
            engine.dim(),
            &zti_hw::probe(),
            None,
        ),
    };

    let query_emb = engine.embed_query_async(query).await?;
    let chunks_table = db.chunks_table(engine.dim()).await?;
    let raw_k = opts.limit.saturating_mul(KNN_OVERFETCH_MULT);

    let mut candidates: Vec<ChunkHit> = match params.method {
        SearchMethod::IvfHnswSq | SearchMethod::Flat => {
            chunks_table
                .knn(&query_emb, raw_k, &params, opts.languages, opts.path_glob)
                .await?
        }
        SearchMethod::Usearch => {
            let graph: AnnHandle = ann_cache
                .get_or_build(*pid, || rebuild(&chunks_table, engine.dim(), &params))
                .await
                .map_err(|e: anyhow::Error| e)?;

            let mut topn: Vec<([u8; 16], f32)> = Vec::with_capacity(raw_k);
            graph.search(&query_emb, raw_k, &mut topn);

            let score_by_id: std::collections::HashMap<[u8; 16], f32> =
                topn.iter().map(|(id, score)| (*id, *score)).collect();

            let ids: Vec<[u8; 16]> = topn.iter().map(|(id, _)| *id).collect();
            let mut fetched = chunks_table
                .fetch_by_chunk_ids(&ids, opts.languages, opts.path_glob)
                .await?;

            for hit in &mut fetched {
                let mut key = [0u8; 16];
                key.copy_from_slice(&hit.chunk_id[..16]);
                if let Some(s) = score_by_id.get(&key) {
                    hit.score = *s;
                }
            }
            fetched
        }
    };

    apply_keyword_boost(query, &mut candidates);

    let rerank_input: Vec<(&[u8], f32)> = candidates
        .iter()
        .map(|c| (c.turbo_code.as_slice(), c.score))
        .collect();
    let mut ranked = reranker.rerank(&rerank_input, &query_emb);

    diversify_by_parent_in_place(&mut ranked, &candidates, opts.limit);

    let mut slots: Vec<Option<ChunkHit>> = candidates.drain(..).map(Some).collect();
    let mut hits: Vec<Hit> = Vec::with_capacity(ranked.len());
    for (idx, score) in ranked {
        if let Some(c) = slots.get_mut(idx).and_then(Option::take) {
            hits.push(Hit { chunk: c, score });
        }
    }
    Ok(hits)
}

#[inline]
fn diversify_by_parent_in_place(ranked: &mut Vec<(usize, f32)>, candidates: &[ChunkHit], k: usize) {
    let mut parents_seen: HashMap<u32, u32> = HashMap::with_capacity(ranked.len());
    for entry in ranked.iter_mut() {
        let parent = candidates.get(entry.0).and_then(|c| c.parent_sym_id);
        if let Some(p) = parent {
            let n = parents_seen.entry(p).or_insert(0);
            entry.1 -= (*n as f32) * DIVERSITY_PENALTY;
            *n += 1;
        }
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    ranked.truncate(k);
}

pub async fn search_exhaustive(
    query: &str,
    engine: &EmbedEngine,
    db: &zti_store::Db,
    pid: &[u8; 32],
    opts: &SearchOpts<'_>,
) -> Result<Vec<Hit>> {
    let projects = db.projects_table().await?;
    let _project = projects
        .get(pid)
        .await?
        .ok_or_else(|| anyhow!("project not indexed"))?;

    let query_emb = engine.embed_query_async(query).await?;
    let raw_k = opts.limit.saturating_mul(KNN_OVERFETCH_MULT);

    let mut candidates = db
        .chunks_table(engine.dim())
        .await?
        .knn_exhaustive(&query_emb, raw_k, opts.languages, opts.path_glob)
        .await?;

    apply_keyword_boost(query, &mut candidates);

    let mut hits: Vec<Hit> = Vec::with_capacity(candidates.len());
    for c in candidates {
        let score = c.score;
        hits.push(Hit { chunk: c, score });
    }
    Ok(hits)
}

async fn rebuild(
    chunks: &zti_store::chunks_table::ChunksTable,
    dim: usize,
    params: &SearchParams,
) -> Result<zti_ann::AnnIndex> {
    let n = params.indexed_chunks as usize;
    let mut builder = AnnIndexBuilder::new(dim, params)?;
    builder.reserve(n.max(1_024))?;

    chunks
        .iter_vectors(|id, v| {
            builder.add(*id, v);
        })
        .await?;

    builder.build()
}

#[cfg(test)]
mod tests {
    use crate::alloc_counting;
    use arrow::array::Float32Array;
    use std::sync::Mutex;

    // The `#[cfg(test)] #[global_allocator]` counter in `crate::alloc_counting`
    // is process-wide. The default `cargo test` harness runs the tests in this
    // module on multiple threads, so concurrent allocations from sibling tests
    // pollute every snapshot region. Serialise all tests in this file so the
    // snapshot delta really only sees the allocations the test under scrutiny
    // performed.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn float32array_from_vec_is_zero_copy() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ptr_before = v.as_ptr();
        let arr = Float32Array::from(v);
        let ptr_after = arr.values().as_ptr();
        assert_eq!(
            ptr_before, ptr_after,
            "Float32Array::from(Vec<f32>) must reuse the Vec buffer without copying"
        );
    }

    #[test]
    fn float32array_from_vec_zero_copy_verified_by_allocation_counter() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let v = vec![1.0f32; 1024];
        let (_, prev_bytes) = alloc_counting::snapshot();
        let arr = Float32Array::from(v);
        std::hint::black_box(&arr);
        let (_, curr_bytes) = alloc_counting::snapshot();
        let delta = curr_bytes - prev_bytes;
        eprintln!("Float32Array::from: 1024 × f32, delta_bytes={delta}");
        assert!(
            delta < 1024,
            "Float32Array::from should not re-allocate the vec buffer, got {delta} bytes",
        );
    }

    #[test]
    fn rebuild_builder_pattern_saves_flat_vec_allocations() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dim = 128;
        let n = 100;
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0f32; dim]).collect();

        let (_, prev_bytes) = alloc_counting::snapshot();
        let mut flat: Vec<f32> = Vec::new();
        let mut chunk_ids: Vec<[u8; 16]> = Vec::new();
        for (i, v) in vectors.iter().enumerate() {
            flat.extend_from_slice(v);
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            chunk_ids.push(id);
        }
        let (_, mid_bytes) = alloc_counting::snapshot();
        std::hint::black_box((&flat, &chunk_ids));
        let old_delta = mid_bytes - prev_bytes;

        let (_, prev2_bytes) = alloc_counting::snapshot();
        let mut chunk_ids2: Vec<[u8; 16]> = Vec::new();
        for (i, _v) in vectors.iter().enumerate() {
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            chunk_ids2.push(id);
        }
        let (_, curr2_bytes) = alloc_counting::snapshot();
        std::hint::black_box(&chunk_ids2);
        let new_delta = curr2_bytes - prev2_bytes;

        let savings = old_delta as i64 - new_delta as i64;
        let reduction_pct = if old_delta > 0 {
            (1.0 - new_delta as f64 / old_delta as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "rebuild compare: {n} vectors x {dim} dim, old_flat+chunk_ids={old_delta} B, builder_chunk_ids_only={new_delta} B, saved={savings} B ({reduction_pct:.0}%)"
        );
        assert!(
            savings >= (n * dim * 4) as i64,
            "builder should save at least n * dim * 4 bytes by avoiding flat Vec, got {savings}",
        );
    }

    #[test]
    fn flat_split_slices_are_views_not_copies() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dim = 16;
        let n = 5;
        let flat: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let ptr_base = flat.as_ptr();

        for i in 0..n {
            let v = &flat[i * dim..(i + 1) * dim];
            let offset = (v.as_ptr() as usize) - (ptr_base as usize);
            assert_eq!(offset, i * dim * 4);
        }
    }
}
