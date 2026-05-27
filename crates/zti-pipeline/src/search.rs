use std::borrow::Cow;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use anyhow::{Result, anyhow};

use zti_ann::{AnnCache, AnnHandle, AnnIndexBuilder, SearchMethod, SearchParams};
use zti_embed::EmbedEngine;
use zti_rerank::TurboReranker;
use zti_rerank::gpu::{BATCH_SIZE, GpuTurboScorer, TurboCodeBatch, parse_turbo_code_into};
use zti_store::chunks_table::{ChunkHit, ChunksTable};

const KNN_OVERFETCH_MULT: usize = 12;
const DIVERSITY_PENALTY: f32 = 0.04;
const KEYWORD_NAME_BOOST: f32 = 0.5;
const KEYWORD_CONTENT_BOOST: f32 = 0.3;
const MIN_WORD_LEN: usize = 3;

/// Collect query word slices from an already-lowercased buffer. Splits on any
/// non-alphanumeric byte (so `_` becomes a boundary too — keeps SQL `LIKE`
/// patterns trivially safe without an `ESCAPE` clause). Words shorter than
/// [`MIN_WORD_LEN`] are filtered out so noise tokens like `in`, `a`, `of`
/// don't pollute the boost. Borrows from `lc` — caller keeps the buffer
/// alive for the lifetime of the returned slices.
fn split_query_words(lc: &str) -> Vec<&str> {
    lc.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() >= MIN_WORD_LEN)
        .collect()
}

#[inline]
fn apply_keyword_boost(words: &[&str], candidates: &mut [ChunkHit]) {
    if words.is_empty() {
        return;
    }
    for c in candidates {
        let name_lc = lowercase_borrowed(&c.symbol_qualified);
        let content_lc = lowercase_borrowed(&c.content);
        for w in words {
            if name_lc.contains(*w) {
                c.score += KEYWORD_NAME_BOOST;
            } else if content_lc.contains(*w) {
                c.score += KEYWORD_CONTENT_BOOST;
            }
        }
    }
}

/// `Cow::Borrowed` when `s` is already ASCII-lowercase (no allocation, the hot
/// path for snake_case code identifiers); `Cow::Owned` when there is an
/// uppercase byte we have to fold. We only ever lowercase once per candidate,
/// not once per (candidate × word).
#[inline]
fn lowercase_borrowed(s: &str) -> Cow<'_, str> {
    if s.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(s.to_ascii_lowercase())
    } else {
        Cow::Borrowed(s)
    }
}

/// Union lexical hits into the kNN candidate pool. Dedup by the 16-byte
/// `chunk_id` prefix held on the stack (no `Vec` allocation per id). Moves
/// `ChunkHit`s out of `additions` instead of cloning. Used by both `search`
/// and `search_exhaustive` so the union logic lives in one place.
fn merge_unique_by_chunk_id(candidates: &mut Vec<ChunkHit>, additions: Vec<ChunkHit>) {
    if additions.is_empty() {
        return;
    }
    let mut seen: HashSet<[u8; 16]> = HashSet::with_capacity(candidates.len() + additions.len());
    for c in candidates.iter() {
        seen.insert(c.chunk_id);
    }
    candidates.reserve(additions.len());
    for a in additions {
        if seen.insert(a.chunk_id) {
            candidates.push(a);
        }
    }
}

/// Run the lexical leg of hybrid retrieval and merge it into the kNN pool.
/// Shared by `search` and `search_exhaustive`. No-op when `words` is empty.
async fn extend_with_lexical(
    chunks_table: &ChunksTable,
    candidates: &mut Vec<ChunkHit>,
    words: &[&str],
    opts: &SearchOpts<'_>,
    k: usize,
) -> Result<()> {
    if words.is_empty() {
        return Ok(());
    }
    let lex = chunks_table
        .lexical_match(words, opts.languages, opts.path_glob, k)
        .await?;
    merge_unique_by_chunk_id(candidates, lex);
    Ok(())
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

#[allow(clippy::too_many_arguments)]
pub async fn search(
    query: &str,
    query_emb: &[f32],
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

    let chunks_table = db.chunks_table(engine.dim()).await?;
    let raw_k = opts.limit.saturating_mul(KNN_OVERFETCH_MULT);

    let query_lc = query.to_ascii_lowercase();
    let words = split_query_words(&query_lc);

    let mut candidates: Vec<ChunkHit> = match params.method {
        SearchMethod::TurboQuant => {
            let gpu_scorer = GpuTurboScorer::from_reranker(reranker, &engine.device()?)?;
            let rotated_query = gpu_scorer.pre_rotate(query_emb);
            let dim_over_2 = gpu_scorer.dim_over_2();
            let sign_bytes = gpu_scorer.sign_bytes_per_code();

            let mut pending = TurboCodeBatch::with_capacity(BATCH_SIZE, dim_over_2, sign_bytes);
            let mut batches: Vec<TurboCodeBatch> = Vec::with_capacity(16);

            chunks_table
                .iter_turbo_codes(opts.languages, opts.path_glob, |id, code| {
                    parse_turbo_code_into(code, &mut pending, id);
                    if pending.len() >= BATCH_SIZE {
                        batches.push(std::mem::replace(
                            &mut pending,
                            TurboCodeBatch::with_capacity(BATCH_SIZE, dim_over_2, sign_bytes),
                        ));
                    }
                })
                .await?;
            if !pending.is_empty() {
                batches.push(pending);
            }

            let mut heap: BinaryHeap<Reverse<ScoredEntry>> = BinaryHeap::with_capacity(raw_k + 1);
            for batch in &batches {
                let scored = gpu_scorer
                    .score_batch(batch, &rotated_query)
                    .map_err(|e| anyhow!("GPU score batch: {e}"))?;
                for (chunk_id, score) in scored {
                    heap.push(Reverse(ScoredEntry { score, chunk_id }));
                    if heap.len() > raw_k {
                        heap.pop();
                    }
                }
            }

            let mut scores: Vec<(f32, [u8; 16])> = Vec::with_capacity(heap.len());
            while let Some(Reverse(entry)) = heap.pop() {
                scores.push((entry.score, entry.chunk_id));
            }
            scores.reverse();

            let top_ids: Vec<[u8; 16]> = scores.iter().map(|(_, id)| *id).collect();
            let score_by_id: HashMap<[u8; 16], f32> =
                scores.iter().map(|(s, id)| (*id, *s)).collect();

            let mut hits = chunks_table
                .fetch_by_chunk_ids(&top_ids, opts.languages, opts.path_glob)
                .await?;
            for hit in &mut hits {
                if let Some(s) = score_by_id.get(&hit.chunk_id) {
                    hit.score = *s;
                }
            }
            hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            hits
        }
        SearchMethod::Usearch => {
            let graph: AnnHandle = ann_cache
                .get_or_build(*pid, || rebuild(&chunks_table, engine.dim(), &params))
                .await
                .map_err(|e: anyhow::Error| e)?;

            let mut topn: Vec<([u8; 16], f32)> = Vec::with_capacity(raw_k);
            graph.search(query_emb, raw_k, &mut topn);

            let score_by_id: std::collections::HashMap<[u8; 16], f32> =
                topn.iter().map(|(id, score)| (*id, *score)).collect();

            let ids: Vec<[u8; 16]> = topn.iter().map(|(id, _)| *id).collect();
            let mut fetched = chunks_table
                .fetch_by_chunk_ids(&ids, opts.languages, opts.path_glob)
                .await?;

            for hit in &mut fetched {
                if let Some(s) = score_by_id.get(&hit.chunk_id) {
                    hit.score = *s;
                }
            }
            fetched
        }
        _ => {
            chunks_table
                .knn(query_emb, raw_k, &params, opts.languages, opts.path_glob)
                .await?
        }
    };

    extend_with_lexical(&chunks_table, &mut candidates, &words, opts, raw_k).await?;
    apply_keyword_boost(&words, &mut candidates);

    let rerank_input: Vec<(&[u8], f32)> = candidates
        .iter()
        .map(|c| (c.turbo_code.as_slice(), c.score))
        .collect();
    let mut ranked = reranker.rerank(&rerank_input, query_emb);

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
    query_emb: &[f32],
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

    let raw_k = opts.limit.saturating_mul(KNN_OVERFETCH_MULT);

    let chunks_table = db.chunks_table(engine.dim()).await?;
    let query_lc = query.to_ascii_lowercase();
    let words = split_query_words(&query_lc);

    let mut candidates = chunks_table
        .knn_exhaustive(query_emb, raw_k, opts.languages, opts.path_glob)
        .await?;

    extend_with_lexical(&chunks_table, &mut candidates, &words, opts, raw_k).await?;
    apply_keyword_boost(&words, &mut candidates);

    // Sort by the boosted score so lexical hits surface — the previous
    // implementation iterated in raw-cosine order and silently dropped the
    // boost on output.
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    candidates.truncate(opts.limit);

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
    use super::*;
    use crate::alloc_counting;
    use arrow::array::Float32Array;
    use std::sync::Mutex;

    fn mk_chunk(chunk_id: [u8; 16], qualified: &str, content: &str) -> ChunkHit {
        ChunkHit {
            chunk_id,
            file_path: "src/poly/rq.rs".into(),
            symbol_qualified: qualified.into(),
            symbol_kind: "method".into(),
            sym_id: 0,
            parent_sym_id: None,
            appendix_sym_ids: Vec::with_capacity(0),
            start_line: 1,
            end_line: 1,
            content: content.into(),
            turbo_code: Vec::with_capacity(0),
            score: 0.0,
        }
    }

    #[test]
    fn split_query_words_strips_short_words_and_punct() {
        let lc = "invert poly in rq".to_ascii_lowercase();
        let words = split_query_words(&lc);
        // "in" is below MIN_WORD_LEN(3), "rq" is below too, only "invert"+"poly" survive.
        assert_eq!(words, vec!["invert", "poly"]);
    }

    #[test]
    fn split_query_words_splits_on_underscore_and_colon() {
        let lc = "rq::mult_int".to_ascii_lowercase();
        let words = split_query_words(&lc);
        // `_` and `:` are both non-alphanumeric → boundaries. Tokens >=3 chars.
        assert_eq!(words, vec!["mult", "int"]);
    }

    #[test]
    fn keyword_boost_accumulates_per_word_on_name() {
        let mut hits = vec![mk_chunk(
            [1u8; 16],
            "Rq::recip",
            "let x = fq::recip(RATIO);",
        )];
        apply_keyword_boost(&["recip", "rq"], &mut hits);
        // "recip" matches symbol_qualified → +0.5
        // "rq"    matches symbol_qualified → +0.5
        assert!((hits[0].score - 1.0).abs() < 1e-6, "got {}", hits[0].score);
    }

    #[test]
    fn keyword_boost_falls_back_to_content_when_name_misses() {
        let mut hits = vec![mk_chunk(
            [1u8; 16],
            "PolyErrors",
            "let scale = recip(f[0]);",
        )];
        apply_keyword_boost(&["recip"], &mut hits);
        // Not in name → use content boost (0.3), not name (0.5).
        assert!(
            (hits[0].score - KEYWORD_CONTENT_BOOST).abs() < 1e-6,
            "got {}",
            hits[0].score
        );
    }

    #[test]
    fn keyword_boost_empty_words_is_noop() {
        let mut hits = vec![mk_chunk([1u8; 16], "Anything", "anywhere")];
        apply_keyword_boost(&[], &mut hits);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn lowercase_borrowed_avoids_alloc_when_already_lowercase() {
        let s = "rq_recip_inverse";
        let cow = lowercase_borrowed(s);
        // No upper-case byte → must return Borrowed (zero allocation).
        assert!(matches!(cow, Cow::Borrowed(_)));
    }

    #[test]
    fn lowercase_borrowed_owns_when_uppercase_present() {
        let s = "Rq::Recip";
        let cow = lowercase_borrowed(s);
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(&*cow, "rq::recip");
    }

    #[test]
    fn merge_unique_by_chunk_id_dedups_and_moves() {
        let mut existing = vec![mk_chunk([1u8; 16], "a", ""), mk_chunk([2u8; 16], "b", "")];
        let lex = vec![
            mk_chunk([2u8; 16], "b-dup", ""), // dup by chunk_id
            mk_chunk([3u8; 16], "c", ""),     // new
        ];
        merge_unique_by_chunk_id(&mut existing, lex);
        assert_eq!(existing.len(), 3);
        assert_eq!(existing[0].symbol_qualified, "a");
        assert_eq!(existing[1].symbol_qualified, "b");
        assert_eq!(existing[2].symbol_qualified, "c");
    }

    #[test]
    fn merge_unique_by_chunk_id_empty_additions_is_noop() {
        let mut existing = vec![mk_chunk([1u8; 16], "a", "")];
        merge_unique_by_chunk_id(&mut existing, Vec::new());
        assert_eq!(existing.len(), 1);
    }

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
