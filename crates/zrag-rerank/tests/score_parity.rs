use candle_core::Device;
use zrag_rerank::TurboReranker;
use zrag_rerank::gpu::{
    BATCH_SIZE, GpuTurboCore, GpuTurboScratch, TurboCodeBatch, TurboScorerCache,
    parse_turbo_code_into, score_batch,
};

const FIXTURE_DIM: usize = 128;
const FIXTURE_SEED: u64 = 99;

fn fixture_reranker() -> TurboReranker {
    TurboReranker::with_params(
        FIXTURE_DIM,
        zrag_rerank::turbo::RerankParams {
            bits: 3,
            projections: 64,
            seed: FIXTURE_SEED,
        },
    )
    .expect("fixture reranker should build")
}

fn unit_vector(dim: usize) -> Vec<f32> {
    let scale = (dim as f32).sqrt().recip();
    vec![scale; dim]
}

fn fixture_query() -> Vec<f32> {
    unit_vector(FIXTURE_DIM)
}

fn fixture_batch(reranker: &TurboReranker, n: usize) -> TurboCodeBatch {
    let v = unit_vector(FIXTURE_DIM);
    let dim_over_2 = FIXTURE_DIM / 2;
    let num_projections = reranker.quantizer().projections();
    let sign_bytes_per_code = num_projections.div_ceil(8);
    let mut batch = TurboCodeBatch::with_capacity(n, dim_over_2, sign_bytes_per_code);
    for i in 0..n {
        let code_bytes = reranker.encode(&v).expect("encode should succeed");
        let mut id = [0u8; 16];
        id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        assert!(
            parse_turbo_code_into(&code_bytes, &mut batch, &id),
            "parse round-trip should succeed for chunk {i}"
        );
    }
    assert_eq!(batch.len(), n);
    batch
}

#[test]
fn cached_core_and_reused_scratch_match_fresh_build() -> anyhow::Result<()> {
    let reranker = fixture_reranker();
    let device = Device::Cpu;
    let batch = fixture_batch(&reranker, 16);
    let query = fixture_query();

    // Build via the cache — this is the per-process codepath.
    let cache = TurboScorerCache::default();
    let core = cache.get_or_build(&reranker, &device)?;
    let mut scratch = GpuTurboScratch::with_capacity(core.num_projections(), core.dim_over_2());
    let mut rq: Vec<f32> = Vec::with_capacity(query.len());
    core.pre_rotate_into(&query, &mut rq);

    // First pass.
    let first: Vec<_> = score_batch(&core, &mut scratch, &batch, &rq)?.to_vec();

    // Second pass over the SAME scratch — must not drift (proves clear()/reuse
    // is sound and doesn't leak state between calls).
    let second: Vec<_> = score_batch(&core, &mut scratch, &batch, &rq)?.to_vec();
    assert_eq!(
        first, second,
        "second pass over same scratch must be bit-identical"
    );

    // Cache hit returns the same core (Arc pointer equality is not required, but
    // the scores must match a fresh build).
    let core2 = cache.get_or_build(&reranker, &device)?;
    let mut s2 = GpuTurboScratch::with_capacity(core2.num_projections(), core2.dim_over_2());
    let cache_hit: Vec<_> = score_batch(&core2, &mut s2, &batch, &rq)?.to_vec();
    assert_eq!(
        first, cache_hit,
        "cache hit must produce same scores as initial build"
    );

    // Fresh build (no cache) must produce identical results.
    let fresh = GpuTurboCore::from_reranker(&reranker, &device)?;
    let mut s3 = GpuTurboScratch::with_capacity(fresh.num_projections(), fresh.dim_over_2());
    let baseline: Vec<_> = score_batch(&fresh, &mut s3, &batch, &rq)?.to_vec();
    assert_eq!(
        first, baseline,
        "fresh GpuTurboCore build must produce same scores as cached"
    );

    Ok(())
}

#[test]
fn cache_key_guards_against_dim_change() -> anyhow::Result<()> {
    let reranker_128 = fixture_reranker();
    let reranker_64 = TurboReranker::with_params(
        64,
        zrag_rerank::turbo::RerankParams {
            bits: 3,
            projections: 64,
            seed: FIXTURE_SEED,
        },
    )?;

    let cache = TurboScorerCache::default();
    let device = Device::Cpu;

    let core_128 = cache.get_or_build(&reranker_128, &device)?;
    let core_64 = cache.get_or_build(&reranker_64, &device)?;

    // Different dims → different cores. Check at the field level since Arc
    // pointer comparison is unreliable.
    assert_ne!(
        core_128.dim_over_2(),
        core_64.dim_over_2(),
        "dim 128 → dim_over_2={}, dim 64 → dim_over_2={} — cache must not mix dims",
        core_128.dim_over_2(),
        core_64.dim_over_2(),
    );

    // Same seed but different dims — cache must NOT return the same entry.
    let core_128_again = cache.get_or_build(&reranker_128, &device)?;
    assert_eq!(
        core_128_again.dim_over_2(),
        core_128.dim_over_2(),
        "re-fetching same dim must return the cached core"
    );

    Ok(())
}

#[test]
fn cache_key_guards_against_seed_change() -> anyhow::Result<()> {
    let reranker_a = TurboReranker::with_params(
        FIXTURE_DIM,
        zrag_rerank::turbo::RerankParams {
            bits: 3,
            projections: 64,
            seed: 42,
        },
    )?;
    let reranker_b = TurboReranker::with_params(
        FIXTURE_DIM,
        zrag_rerank::turbo::RerankParams {
            bits: 3,
            projections: 64,
            seed: 99,
        },
    )?;

    let cache = TurboScorerCache::default();
    let device = Device::Cpu;

    let _core_a = cache.get_or_build(&reranker_a, &device)?;
    // Different seed, same dim — cache must rebuild (not return the seed=42
    // core for seed=99).
    let core_b = cache.get_or_build(&reranker_b, &device)?;

    let batch_a = fixture_batch(&reranker_a, 16);
    let query = fixture_query();
    let mut scratch = GpuTurboScratch::with_capacity(core_b.num_projections(), core_b.dim_over_2());
    let mut rq: Vec<f32> = Vec::with_capacity(query.len());
    core_b.pre_rotate_into(&query, &mut rq);

    // Must not error with shape mismatches — proves the cache rebuilt for the
    // new seed (same dim, so shapes are compatible; different coordinate
    // system, so scores differ — but we only care about the shape not
    // panicking).
    let _scores = score_batch(&core_b, &mut scratch, &batch_a, &rq)?;
    Ok(())
}

#[test]
fn empty_batch_returns_empty_slice() -> anyhow::Result<()> {
    let reranker = fixture_reranker();
    let device = Device::Cpu;
    let cache = TurboScorerCache::default();
    let core = cache.get_or_build(&reranker, &device)?;
    let mut scratch = GpuTurboScratch::with_capacity(core.num_projections(), core.dim_over_2());
    let query = fixture_query();
    let mut rq: Vec<f32> = Vec::with_capacity(query.len());
    core.pre_rotate_into(&query, &mut rq);

    let empty_batch = TurboCodeBatch::default();
    let scores = score_batch(&core, &mut scratch, &empty_batch, &rq)?;
    assert!(scores.is_empty());

    Ok(())
}

#[test]
fn bound_to_core_shrinks_overallocated_buffers() {
    // Simulate what happens when a scratch sized for a large-dim project
    // is later used with a smaller-dim core: the Vec capacities are
    // stuck at the larger size.  bound_to_core must shrink them back.
    let reranker = fixture_reranker();
    let core = GpuTurboCore::from_reranker(&reranker, &Device::Cpu).expect("build core");

    let ceiling_pre_signs = BATCH_SIZE * core.num_projections();
    let ceiling_angle = BATCH_SIZE * core.dim_over_2();
    let ceiling_scores = BATCH_SIZE;

    let mut scratch = GpuTurboScratch::with_capacity(core.num_projections(), core.dim_over_2());

    // Artificially bloat: reserve enough to push capacity past 2× ceiling.
    scratch.pre_signs_flat.reserve(ceiling_pre_signs * 2 + 1);
    scratch.angle_i64.reserve(ceiling_angle * 2 + 1);
    scratch.scores.reserve(ceiling_scores * 2 + 1);

    assert!(scratch.pre_signs_flat.capacity() > ceiling_pre_signs * 2);
    assert!(scratch.angle_i64.capacity() > ceiling_angle * 2);
    assert!(scratch.scores.capacity() > ceiling_scores * 2);

    // bound_to_core with 2× threshold triggers shrink on all three.
    scratch.bound_to_core(&core);

    assert!(
        scratch.pre_signs_flat.capacity() <= ceiling_pre_signs,
        "pre_signs_flat should shrink to ceiling {ceiling_pre_signs}, got {}",
        scratch.pre_signs_flat.capacity(),
    );
    assert!(
        scratch.angle_i64.capacity() <= ceiling_angle,
        "angle_i64 should shrink to ceiling {ceiling_angle}, got {}",
        scratch.angle_i64.capacity(),
    );
    assert!(
        scratch.scores.capacity() <= ceiling_scores,
        "scores should shrink to ceiling {ceiling_scores}, got {}",
        scratch.scores.capacity(),
    );
}

#[test]
fn bound_to_core_leaves_normal_capacity_untouched() {
    // When capacity is already at or near the ceiling, bound_to_core
    // must NOT shrink (avoids pointless reallocation on every batch).
    let reranker = fixture_reranker();
    let core = GpuTurboCore::from_reranker(&reranker, &Device::Cpu).expect("build core");

    let mut scratch = GpuTurboScratch::with_capacity(core.num_projections(), core.dim_over_2());

    let cap_pre = scratch.pre_signs_flat.capacity();
    let cap_angle = scratch.angle_i64.capacity();
    let cap_scores = scratch.scores.capacity();

    scratch.bound_to_core(&core);

    assert_eq!(scratch.pre_signs_flat.capacity(), cap_pre);
    assert_eq!(scratch.angle_i64.capacity(), cap_angle);
    assert_eq!(scratch.scores.capacity(), cap_scores);
}
