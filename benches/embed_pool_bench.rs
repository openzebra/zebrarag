use candle_core::{DType, Device, Tensor};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use zti_embed::normalize::normalize_l2;
use zti_embed::pooling::{PoolingStrategy, pool_on_device, pool_row_into};

const BATCH: usize = 32;
const SEQ: usize = 256;
const DIM: usize = 384;

fn device() -> (Device, &'static str) {
    #[cfg(target_os = "macos")]
    if let Ok(d) = Device::new_metal(0) {
        return (d, "Metal");
    }
    (Device::Cpu, "Cpu")
}

/// OLD path: copy the whole (b,s,dim) tensor to host, then pool+normalize per
/// row. Each row's temp lives on the stack (`[f32; DIM]`); no indexing.
fn pool_cpu_readback(output: &Tensor, strat: &PoolingStrategy) -> anyhow::Result<Vec<f32>> {
    let flat = output
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(BATCH * DIM);
    for row in flat.chunks_exact(SEQ * DIM) {
        let mut tmp = [0f32; DIM];
        pool_row_into(strat, row, SEQ, &mut tmp);
        normalize_l2(&mut tmp);
        out.extend_from_slice(&tmp);
    }
    Ok(out)
}

fn run(c: &mut Criterion) -> anyhow::Result<()> {
    let (dev, label) = device();
    let output = Tensor::randn(0f32, 1f32, (BATCH, SEQ, DIM), &dev)?;
    let mask = Tensor::ones((BATCH, SEQ), DType::F32, &dev)?;
    let strat = PoolingStrategy::Mean;

    let on_dev = pool_on_device(&output, &mask, &strat, BATCH)?;
    let cpu = pool_cpu_readback(&output, &strat)?;
    if on_dev.iter().zip(&cpu).any(|(x, y)| (x - y).abs() >= 1e-4) {
        anyhow::bail!("on-device pool diverged from CPU readback path");
    }

    c.bench_function(
        &format!("embed_pool/on_device_{label}_b{BATCH}_s{SEQ}_d{DIM}"),
        |bch| {
            bch.iter(|| pool_on_device(black_box(&output), black_box(&mask), &strat, BATCH));
        },
    );
    c.bench_function(
        &format!("embed_pool/cpu_readback_{label}_b{BATCH}_s{SEQ}_d{DIM}"),
        |bch| {
            bch.iter(|| pool_cpu_readback(black_box(&output), &strat));
        },
    );
    Ok(())
}

fn bench_pool(c: &mut Criterion) {
    if let Err(e) = run(c) {
        eprintln!("embed_pool bench skipped: {e}");
    }
}

criterion_group!(benches, bench_pool);
criterion_main!(benches);
