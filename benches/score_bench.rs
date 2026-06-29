use candle_core::Device;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use zrag_rerank::TurboReranker;
use zrag_rerank::gpu::{
    GpuTurboCore, GpuTurboScratch, TurboCodeBatch, TurboScorerCache, parse_turbo_code_into,
    score_batch,
};

const BENCH_DIM: usize = 128;
const BENCH_SEED: u64 = 42;
const BATCH_N: usize = 4096;
const BENCH_DEVICE: Device = Device::Cpu;
const DEVICE_LABEL: &str = "Cpu";

fn bench_name(base: &str) -> String {
    format!("{base}/n={BATCH_N}_dim={BENCH_DIM}_{DEVICE_LABEL}")
}

fn bench_reranker() -> TurboReranker {
    TurboReranker::with_params(
        BENCH_DIM,
        zrag_rerank::turbo::RerankParams {
            bits: 3,
            projections: 0,
            seed: BENCH_SEED,
        },
    )
    .expect("bench reranker")
}

fn bench_query() -> Vec<f32> {
    let scale = (BENCH_DIM as f32).sqrt().recip();
    vec![scale; BENCH_DIM]
}

fn bench_batch(reranker: &TurboReranker) -> TurboCodeBatch {
    let v = bench_query();
    let dim_over_2 = BENCH_DIM / 2;
    let num_projections = reranker.quantizer().projections();
    let sign_bytes_per_code = num_projections.div_ceil(8);
    let mut batch = TurboCodeBatch::with_capacity(BATCH_N, dim_over_2, sign_bytes_per_code);
    for i in 0..BATCH_N {
        let code_bytes = reranker.encode(&v).expect("encode");
        let mut id = [0u8; 16];
        id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        parse_turbo_code_into(&code_bytes, &mut batch, &id);
    }
    batch
}

fn bench_scorer_construction(c: &mut Criterion) {
    let reranker = bench_reranker();

    c.bench_function(&bench_name("scorer_construction"), |b| {
        b.iter(|| {
            // This is the per-query cost BEFORE caching — LU + RNG + tensor
            // uploads. With the cache, this runs once (cold), then never again.
            let core = GpuTurboCore::from_reranker(black_box(&reranker), black_box(&BENCH_DEVICE))
                .expect("build core");
            black_box(core);
        });
    });
}

fn bench_scorer_cache_hit(c: &mut Criterion) {
    let reranker = bench_reranker();
    let cache = TurboScorerCache::default();

    cache
        .get_or_build(&reranker, &BENCH_DEVICE)
        .expect("cache fill");

    c.bench_function(&bench_name("scorer_cache_hit"), |b| {
        b.iter(|| {
            let core = cache
                .get_or_build(black_box(&reranker), black_box(&BENCH_DEVICE))
                .expect("cache hit");
            black_box(core);
        });
    });
}

fn bench_score_batch(c: &mut Criterion) {
    let reranker = bench_reranker();
    let cache = TurboScorerCache::default();
    let core = cache
        .get_or_build(&reranker, &BENCH_DEVICE)
        .expect("build core");
    let batch = bench_batch(&reranker);
    let query = bench_query();
    let mut rq: Vec<f32> = Vec::with_capacity(query.len());
    core.pre_rotate_into(&query, &mut rq);

    let mut scratch = GpuTurboScratch::with_capacity(core.num_projections(), core.dim_over_2());

    c.bench_function(&bench_name("score_batch"), |b| {
        b.iter(|| {
            let scores = score_batch(
                black_box(&core),
                black_box(&mut scratch),
                black_box(&batch),
                black_box(&rq),
            )
            .expect("score_batch");
            black_box(scores);
        });
    });
}

criterion_group!(
    benches,
    bench_scorer_construction,
    bench_scorer_cache_hit,
    bench_score_batch,
);
criterion_main!(benches);
