use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{Result, anyhow};

use zti_ann::{AnnCache, AnnHandle, AnnIndexBuilder, SearchMethod, SearchParams};
use zti_common::file_type::FileType;
use zti_embed::AnyEmbedEngine;
use zti_rerank::TurboReranker;
use zti_rerank::gpu::{
    BATCH_SIZE, GpuTurboScratch, TurboCodeBatch, TurboScorerCache, parse_turbo_code_into,
    score_batch,
};
use zti_store::chunks_table::{ChunkHit, ChunksTable};

#[cfg(test)]
use zti_common::chunk_strategy::ChunkStrategy;

const KNN_OVERFETCH_MULT: usize = 12;
const DIVERSITY_PENALTY: f32 = 0.04;

#[inline]
const fn file_type_rank(ft: FileType) -> u8 {
    match ft {
        FileType::Source => 0,
        FileType::Config => 1,
        FileType::Doc => 2,
        FileType::Test => 3,
    }
}

#[inline]
fn push_scored(
    heap: &mut BinaryHeap<Reverse<ScoredEntry>>,
    scored: &[([u8; 16], f32)],
    raw_k: usize,
) {
    for &(chunk_id, score) in scored {
        heap.push(Reverse(ScoredEntry { score, chunk_id }));
        if heap.len() > raw_k {
            heap.pop();
        }
    }
}

pub struct SearchOpts<'a> {
    pub limit: usize,
    pub languages: Option<&'a [String]>,
    pub path_glob: Option<&'a str>,
    pub include_tests: bool,
}

pub struct Hit {
    pub chunk: ChunkHit,
    pub score: f32,
}

pub struct SearchOutcome {
    pub hits: Vec<Hit>,
}

#[allow(clippy::too_many_arguments)]
async fn run_vector_leg(
    chunks_table: &ChunksTable,
    engine: &AnyEmbedEngine,
    reranker: &TurboReranker,
    ann_cache: &AnnCache,
    turbo_cache: &TurboScorerCache,
    hardware: &zti_hw::Hardware,
    pid: &[u8; 32],
    params: &SearchParams,
    query_emb: &[f32],
    raw_k: usize,
    opts: &SearchOpts<'_>,
) -> Result<Vec<ChunkHit>> {
    match params.method {
        SearchMethod::TurboQuant => {
            let device = engine.device_with_hardware(hardware)?;
            let core = turbo_cache.get_or_build(reranker, &device)?;
            let mut scratch =
                GpuTurboScratch::with_capacity(core.num_projections(), core.dim_over_2());
            let mut rotated_query: Vec<f32> = Vec::with_capacity(engine.dim());
            core.pre_rotate_into(query_emb, &mut rotated_query);

            let mut batch = TurboCodeBatch::with_capacity(
                BATCH_SIZE,
                core.dim_over_2(),
                core.sign_bytes_per_code(),
            );
            let mut heap: BinaryHeap<Reverse<ScoredEntry>> = BinaryHeap::with_capacity(raw_k + 1);

            chunks_table
                .iter_turbo_codes(
                    opts.languages,
                    opts.path_glob,
                    opts.include_tests,
                    |id, code| {
                        parse_turbo_code_into(code, &mut batch, id);
                        if batch.len() >= BATCH_SIZE {
                            let scored = tokio::task::block_in_place(|| {
                                score_batch(&core, &mut scratch, &batch, &rotated_query)
                                    .map_err(|e| anyhow!("GPU score batch: {e}"))
                            })?;
                            push_scored(&mut heap, scored, raw_k);
                            batch.clear();
                        }
                        Ok(true)
                    },
                )
                .await?;
            if !batch.is_empty() {
                let scored = tokio::task::block_in_place(|| {
                    score_batch(&core, &mut scratch, &batch, &rotated_query)
                        .map_err(|e| anyhow!("GPU score batch: {e}"))
                })?;
                push_scored(&mut heap, scored, raw_k);
            }

            let mut scores: Vec<(f32, [u8; 16])> = Vec::with_capacity(heap.len());
            while let Some(Reverse(entry)) = heap.pop() {
                scores.push((entry.score, entry.chunk_id));
            }
            scores.reverse();

            let top_ids: Vec<[u8; 16]> = scores.iter().map(|(_, id)| *id).collect();
            let mut score_by_id: HashMap<[u8; 16], f32> = HashMap::with_capacity(scores.len());
            for (score, id) in &scores {
                score_by_id.insert(*id, *score);
            }

            let mut hits = chunks_table
                .fetch_by_chunk_ids(&top_ids, opts.languages, opts.path_glob, opts.include_tests)
                .await?;
            for hit in &mut hits {
                if let Some(score) = score_by_id.get(&hit.chunk_id) {
                    hit.score = *score;
                }
            }
            hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            Ok(hits)
        }
        SearchMethod::Usearch => {
            let graph: AnnHandle = ann_cache
                .get_or_build(*pid, || rebuild(chunks_table, engine.dim(), params))
                .await
                .map_err(|e: anyhow::Error| e)?;

            let mut topn: Vec<([u8; 16], f32)> = Vec::with_capacity(raw_k);
            graph.search(query_emb, raw_k, &mut topn);

            let mut score_by_id: HashMap<[u8; 16], f32> = HashMap::with_capacity(topn.len());
            for (id, score) in &topn {
                score_by_id.insert(*id, *score);
            }

            let ids: Vec<[u8; 16]> = topn.iter().map(|(id, _)| *id).collect();
            let mut fetched = chunks_table
                .fetch_by_chunk_ids(&ids, opts.languages, opts.path_glob, opts.include_tests)
                .await?;

            for hit in &mut fetched {
                if let Some(score) = score_by_id.get(&hit.chunk_id) {
                    hit.score = *score;
                }
            }
            Ok(fetched)
        }
        _ => {
            chunks_table
                .knn(
                    query_emb,
                    raw_k,
                    params,
                    opts.languages,
                    opts.path_glob,
                    opts.include_tests,
                )
                .await
        }
    }
}

fn materialize_fused_hits(
    vec_hits: Vec<ChunkHit>,
    lex_hits: Vec<ChunkHit>,
    raw_k: usize,
) -> Vec<ChunkHit> {
    let vec_ids: Vec<[u8; 16]> = vec_hits.iter().map(|hit| hit.chunk_id).collect();
    let lex_ids: Vec<[u8; 16]> = lex_hits.iter().map(|hit| hit.chunk_id).collect();
    let fused = crate::fusion::rrf(&[&vec_ids, &lex_ids], raw_k);

    let mut by_id: HashMap<[u8; 16], ChunkHit> =
        HashMap::with_capacity(vec_hits.len() + lex_hits.len());
    vec_hits.into_iter().chain(lex_hits).for_each(|hit| {
        by_id.entry(hit.chunk_id).or_insert(hit);
    });

    let mut candidates = Vec::with_capacity(fused.len());
    for (id, rrf_score) in fused {
        if let Some(mut hit) = by_id.remove(&id) {
            hit.score = rrf_score;
            candidates.push(hit);
        }
    }
    candidates
}

#[allow(clippy::too_many_arguments)]
pub async fn search(
    query: &str,
    query_emb: &[f32],
    engine: &AnyEmbedEngine,
    db: &zti_store::Db,
    reranker: &TurboReranker,
    ann_cache: &AnnCache,
    turbo_cache: &TurboScorerCache,
    hardware: &zti_hw::Hardware,
    pid: &[u8; 32],
    opts: &SearchOpts<'_>,
    params_override: Option<&SearchParams>,
    total_chunks_override: Option<usize>,
) -> Result<SearchOutcome> {
    let project = if params_override.is_some() && total_chunks_override.is_some() {
        None
    } else {
        Some(
            db.projects_table()
                .await?
                .get(pid)
                .await?
                .ok_or_else(|| anyhow!("project not indexed"))?,
        )
    };

    let parsed_params: Option<SearchParams> = if params_override.is_some() {
        None
    } else {
        project
            .as_ref()
            .and_then(|row| row.search_params.as_deref())
            .and_then(|params| toml::from_str(params).ok())
    };
    let fallback_params;
    let params = match params_override.or(parsed_params.as_ref()) {
        Some(params) => params,
        None => {
            fallback_params = zti_ann::choose_method(
                total_chunks_override
                    .or_else(|| project.as_ref().map(|row| row.total_chunks as usize))
                    .ok_or_else(|| anyhow!("project not indexed"))?,
                engine.dim(),
                hardware,
                None,
            );
            &fallback_params
        }
    };

    let chunks_table = db.chunks_table(engine.dim()).await?;
    let total_chunks = total_chunks_override
        .or_else(|| project.as_ref().map(|row| row.total_chunks as usize))
        .ok_or_else(|| anyhow!("project not indexed"))?;
    let overfetch = if total_chunks > 100_000 {
        6
    } else if total_chunks > 20_000 {
        8
    } else {
        KNN_OVERFETCH_MULT
    };
    let raw_k = opts.limit.saturating_mul(overfetch);

    let (vec_res, lex_res) = tokio::join!(
        run_vector_leg(
            &chunks_table,
            engine,
            reranker,
            ann_cache,
            turbo_cache,
            hardware,
            pid,
            params,
            query_emb,
            raw_k,
            opts,
        ),
        chunks_table.lexical_match(
            query,
            opts.languages,
            opts.path_glob,
            opts.include_tests,
            raw_k,
        ),
    );
    let vec_hits = vec_res?;
    let lex_hits = lex_res?;
    let mut candidates = materialize_fused_hits(vec_hits, lex_hits, raw_k);

    let rerank_input: Vec<(&[u8], f32)> = candidates
        .iter()
        .map(|candidate| (candidate.turbo_code.as_slice(), candidate.score))
        .collect();
    let mut ranked = tokio::task::block_in_place(|| reranker.rerank(&rerank_input, query_emb));

    // Dedup BEFORE diversify so the overfetch pool (raw_k = limit × overfetch)
    // absorbs duplicate-symbol drops; diversify then makes the final exact-
    // `limit` cut. The reverse order truncates to `limit` first and lets dedup
    // shrink below it with no refill (the cause of under-filled result pages).
    dedup_by_symbol_in_place(&mut ranked, &candidates);
    diversify_by_parent_in_place(&mut ranked, &candidates, opts.limit);

    let mut slots: Vec<Option<ChunkHit>> = candidates.drain(..).map(Some).collect();
    let mut hits: Vec<Hit> = Vec::with_capacity(ranked.len());
    for (idx, score) in ranked {
        if let Some(chunk) = slots.get_mut(idx).and_then(Option::take) {
            hits.push(Hit { chunk, score });
        }
    }
    Ok(SearchOutcome { hits })
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
    ranked.sort_by(|a, b| {
        let ta = candidates
            .get(a.0)
            .map_or(0, |c| file_type_rank(c.file_type));
        let tb = candidates
            .get(b.0)
            .map_or(0, |c| file_type_rank(c.file_type));
        ta.cmp(&tb)
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
    });
    ranked.truncate(k);
}

/// Drop repeat chunks of the same symbol so one symbol can't crowd the
/// results. Chunks whose `sym_id` is `u32::MAX` (the "no owning symbol"
/// sentinel used by PDF pages, text files, and TSV/PSV rows) are each distinct
/// passages and are kept unconditionally — deduping them would collapse an
/// entire document to a single hit.
#[inline]
fn dedup_by_symbol_in_place(ranked: &mut Vec<(usize, f32)>, candidates: &[ChunkHit]) {
    let mut seen_sym_ids: HashSet<u32> = HashSet::with_capacity(ranked.len());
    ranked.retain(|(idx, _)| {
        let sym_id = candidates[*idx].sym_id;
        sym_id == u32::MAX || seen_sym_ids.insert(sym_id)
    });
}

pub async fn search_exhaustive(
    query: &str,
    query_emb: &[f32],
    engine: &AnyEmbedEngine,
    db: &zti_store::Db,
    pid: &[u8; 32],
    opts: &SearchOpts<'_>,
) -> Result<SearchOutcome> {
    let projects = db.projects_table().await?;
    let _project = projects
        .get(pid)
        .await?
        .ok_or_else(|| anyhow!("project not indexed"))?;

    let raw_k = opts.limit.saturating_mul(KNN_OVERFETCH_MULT);
    let chunks_table = db.chunks_table(engine.dim()).await?;

    let (vec_res, lex_res) = tokio::join!(
        chunks_table.knn_exhaustive(
            query_emb,
            raw_k,
            opts.languages,
            opts.path_glob,
            opts.include_tests,
        ),
        chunks_table.lexical_match(
            query,
            opts.languages,
            opts.path_glob,
            opts.include_tests,
            raw_k,
        ),
    );
    let vec_hits = vec_res?;
    let mut candidates = materialize_fused_hits(vec_hits, lex_res?, raw_k);
    candidates.sort_by(|a, b| {
        file_type_rank(a.file_type)
            .cmp(&file_type_rank(b.file_type))
            .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal))
    });

    let mut hits: Vec<Hit> = Vec::with_capacity(candidates.len().min(opts.limit));
    candidates.into_iter().take(opts.limit).for_each(|chunk| {
        let score = chunk.score;
        hits.push(Hit { chunk, score });
    });
    Ok(SearchOutcome { hits })
}

async fn rebuild(
    chunks: &zti_store::chunks_table::ChunksTable,
    dim: usize,
    params: &SearchParams,
) -> Result<zti_ann::AnnIndex> {
    let actual = chunks.len().await?;
    let mut builder = AnnIndexBuilder::new(dim, params)?;
    builder.reserve(actual.max(1_024))?;

    chunks
        .iter_vectors(|id, v| {
            builder.add(*id, v);
        })
        .await?;

    builder.build()
}

#[derive(Copy, Clone)]
struct ScoredEntry {
    score: f32,
    chunk_id: [u8; 16],
}

impl PartialEq for ScoredEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredEntry {}

impl PartialOrd for ScoredEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
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

#[cfg(test)]
mod tq_tests {
    use super::*;

    fn make_candidate(parent_sym_id: Option<u32>) -> ChunkHit {
        make_candidate_with_file_type(parent_sym_id, FileType::Source)
    }

    fn make_candidate_with_file_type(parent_sym_id: Option<u32>, file_type: FileType) -> ChunkHit {
        make_candidate_with_sym(parent_sym_id, file_type, 0)
    }

    fn make_candidate_with_sym(
        parent_sym_id: Option<u32>,
        file_type: FileType,
        sym_id: u32,
    ) -> ChunkHit {
        ChunkHit {
            chunk_id: [0u8; 16],
            file_path: String::new(),
            file_type,
            symbol_qualified: String::new(),
            symbol_kind: String::new(),
            sym_id,
            sub_chunk_idx: 0,
            total_sub_chunks: 1,
            chunk_strategy: ChunkStrategy::Symbol,
            parent_sym_id,
            appendix_sym_ids: Vec::with_capacity(0),
            start_line: 0,
            end_line: 0,
            content: String::new(),
            turbo_code: Vec::with_capacity(0),
            score: 0.0,
        }
    }

    #[test]
    fn diversify_penalizes_repeated_parent() {
        let candidates = vec![
            make_candidate(Some(1)),
            make_candidate(Some(1)),
            make_candidate(Some(2)),
        ];
        let mut ranked: Vec<(usize, f32)> = vec![(0, 10.0), (1, 9.0), (2, 8.0)];
        diversify_by_parent_in_place(&mut ranked, &candidates, 3);
        assert!(ranked.len() == 3);
        let pen = DIVERSITY_PENALTY;
        assert!(
            (ranked[0].1 - 10.0).abs() < 1e-6,
            "first occurrence of parent=1 should have no penalty, got {}",
            ranked[0].1
        );
        assert!(
            (ranked[1].1 - (9.0 - pen)).abs() < 1e-6,
            "second occurrence of parent=1 should be penalized by {pen}, got {}",
            ranked[1].1
        );
        assert!(
            (ranked[2].1 - 8.0).abs() < 1e-6,
            "parent=2 has no repeats, should be unpenalized, got {}",
            ranked[2].1
        );
    }

    #[test]
    fn diversify_truncates_to_k() {
        let candidates = vec![make_candidate(None); 10];
        let mut ranked: Vec<(usize, f32)> = (0..10).map(|i| (i, i as f32)).collect();
        diversify_by_parent_in_place(&mut ranked, &candidates, 3);
        assert_eq!(ranked.len(), 3);
    }

    #[test]
    fn diversify_no_parent_noop() {
        let candidates = vec![make_candidate(None), make_candidate(None)];
        let mut ranked: Vec<(usize, f32)> = vec![(0, 5.0), (1, 3.0)];
        diversify_by_parent_in_place(&mut ranked, &candidates, 5);
        assert_eq!(ranked.len(), 2);
        assert!((ranked[0].1 - 5.0).abs() < 1e-6);
        assert!((ranked[1].1 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn diversify_empty_input() {
        let candidates: Vec<ChunkHit> = Vec::with_capacity(0);
        let mut ranked: Vec<(usize, f32)> = Vec::with_capacity(0);
        diversify_by_parent_in_place(&mut ranked, &candidates, 5);
        assert!(ranked.is_empty());
    }

    #[test]
    fn diversify_orders_source_before_doc_for_equal_scores() {
        let candidates = vec![
            make_candidate_with_file_type(None, FileType::Doc),
            make_candidate_with_file_type(None, FileType::Source),
        ];
        let mut ranked: Vec<(usize, f32)> = vec![(0, 1.0), (1, 1.0)];
        diversify_by_parent_in_place(&mut ranked, &candidates, 2);
        assert_eq!(ranked.first().map(|entry| entry.0), Some(1));
    }

    #[test]
    fn dedup_drops_repeat_chunks_of_same_symbol() {
        // Two chunks of symbol 7 → only the first survives.
        let candidates = vec![
            make_candidate_with_sym(None, FileType::Source, 7),
            make_candidate_with_sym(None, FileType::Source, 7),
        ];
        let mut ranked: Vec<(usize, f32)> = vec![(0, 1.0), (1, 0.9)];
        dedup_by_symbol_in_place(&mut ranked, &candidates);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, 0);
    }

    #[test]
    fn dedup_keeps_all_sentinel_chunks_regression() {
        // u32::MAX marks "no owning symbol" (PDF pages, text, TSV rows). Every
        // such chunk is a distinct passage and MUST survive — deduping them
        // would collapse a whole document to one search hit.
        let candidates = vec![
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
        ];
        let mut ranked: Vec<(usize, f32)> = vec![(0, 1.0), (1, 0.9), (2, 0.8)];
        dedup_by_symbol_in_place(&mut ranked, &candidates);
        assert_eq!(ranked.len(), 3);
    }

    #[test]
    fn dedup_mixed_symbols_and_sentinels() {
        // Real symbols dedup; sentinels survive; distinct symbols survive.
        let candidates = vec![
            make_candidate_with_sym(None, FileType::Source, 1),
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
            make_candidate_with_sym(None, FileType::Source, 1), // dup of #0
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
            make_candidate_with_sym(None, FileType::Source, 2), // distinct
        ];
        let mut ranked: Vec<(usize, f32)> = vec![(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)];
        dedup_by_symbol_in_place(&mut ranked, &candidates);
        // Drops only idx 2 (dup of symbol 1). idx 0,1,3,4 survive.
        assert_eq!(ranked.len(), 4);
        assert!(ranked.iter().all(|(idx, _)| *idx != 2));
    }

    #[test]
    fn dedup_before_diversify_reaches_limit() {
        // Regression for the under-filled result page: an overfetched pool
        // with duplicate symbols must still reach exactly `limit` after
        // dedup-then-diversify, because dedup runs on the full pool (not the
        // truncated one) and diversify makes the final cut.
        let candidates = vec![
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
            make_candidate_with_sym(None, FileType::Source, 7),
            make_candidate_with_sym(None, FileType::Source, 7), // dup
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
            make_candidate_with_sym(None, FileType::Source, 8),
            make_candidate_with_sym(None, FileType::Doc, u32::MAX),
        ];
        let mut ranked: Vec<(usize, f32)> =
            vec![(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5), (6, 0.4)];
        // Compose in the fixed order (dedup then diversify).
        dedup_by_symbol_in_place(&mut ranked, &candidates);
        diversify_by_parent_in_place(&mut ranked, &candidates, 5);
        assert_eq!(ranked.len(), 5);
    }
}
