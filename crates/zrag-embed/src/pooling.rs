use candle_core::{D, DType, Device, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    Mean,
    Cls,
}

/// Pool a single row from a contiguous `(seq * dim)` slice into a caller-owned
/// `dim`-length buffer. Mean pooling respects the attention mask via `valid`;
/// CLS takes index 0. Zero allocation on the hot path: the caller owns the
/// destination and we just write into it.
pub fn pool_row_into(strategy: &PoolingStrategy, data: &[f32], valid: usize, out: &mut [f32]) {
    let dim = out.len();
    match strategy {
        PoolingStrategy::Mean => {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            if valid == 0 {
                return;
            }
            for row in data.chunks_exact(dim).take(valid) {
                for (s, &v) in out.iter_mut().zip(row) {
                    *s += v;
                }
            }
            let c = (valid as f32).recip();
            for x in out.iter_mut() {
                *x *= c;
            }
        }
        PoolingStrategy::Cls => {
            if let Some(row) = data.get(..dim) {
                out.copy_from_slice(row);
            }
        }
    }
}

/// Pool `(batch, seq, dim)` hidden states over the seq axis and L2-normalize,
/// entirely on the tensor's device, then read back only the `(real_batch, dim)`
/// result. Numerically equivalent to `pool_row_into` + `normalize_l2` applied
/// row-wise to a full CPU readback, but the large `(batch*seq*dim)` tensor never
/// leaves the GPU — the host copy shrinks by the `seq` factor.
///
/// `mask` is the `(batch, seq)` attention mask (1 = real token, 0 = pad); for
/// mean pooling its row-sum equals the old `valid_counts`, so results match.
///
/// Allocation budget: every line is a device-side tensor op (storage owned by
/// the backend); the single host allocation is the returned `(real_batch*dim)`
/// `Vec` from `to_vec1` — essential, since it is moved into `Pooled.data` and
/// its length is a runtime value (cannot be a stack `[T; N]`). No clones.
///
/// # Errors
///
/// Returns any tensor backend error from dtype conversion, matmul/reduction,
/// normalization, device transfer, flattening, or host extraction.
pub fn pool_on_device(
    output: &Tensor,
    mask: &Tensor,
    strategy: &PoolingStrategy,
    real_batch: usize,
) -> Result<Vec<f32>> {
    let pooled = match strategy {
        PoolingStrategy::Mean => {
            let maskf = mask.to_dtype(DType::F32)?;
            let summed = if output.dtype() == DType::F32 {
                maskf.unsqueeze(1)?.matmul(output)?.squeeze(1)?
            } else {
                mask.to_dtype(output.dtype())?
                    .unsqueeze(1)?
                    .matmul(output)?
                    .squeeze(1)?
                    .to_dtype(DType::F32)?
            };
            let counts = maskf.sum_keepdim(1)?.clamp(1f32, f32::INFINITY)?;
            summed.broadcast_div(&counts)?
        }
        PoolingStrategy::Cls => output.narrow(1, 0, 1)?.squeeze(1)?.to_dtype(DType::F32)?,
    };

    let norm = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let pooled = pooled.broadcast_div(&norm.clamp(f32::EPSILON, f32::INFINITY)?)?;
    let pooled = pooled.narrow(0, 0, real_batch)?.to_device(&Device::Cpu)?;
    pooled.flatten_all()?.to_vec1::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize_l2;
    use candle_core::{Device, Tensor};

    const B: usize = 3;
    const S: usize = 7;
    const DIM: usize = 5;

    /// Old path: full readback, then pool+normalize each row on the CPU.
    fn cpu_reference(
        output: &Tensor,
        valid: &[usize],
        strat: &PoolingStrategy,
    ) -> anyhow::Result<Vec<f32>> {
        let flat = output.flatten_all()?.to_vec1::<f32>()?;
        let mut out = Vec::with_capacity(B * DIM);
        for (row, &v) in flat.chunks_exact(S * DIM).zip(valid) {
            let mut tmp = [0f32; DIM];
            pool_row_into(strat, row, v, &mut tmp);
            normalize_l2(&mut tmp);
            out.extend_from_slice(&tmp);
        }
        Ok(out)
    }

    #[test]
    fn on_device_matches_cpu_reference() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let output = Tensor::randn(0f32, 1f32, (B, S, DIM), &dev)?;
        let valid: [usize; B] = std::array::from_fn(|i| (i + 2).min(S));
        let mut mask = [0f32; B * S];
        for (i, &v) in valid.iter().enumerate() {
            for slot in mask.iter_mut().skip(i * S).take(v) {
                *slot = 1.0;
            }
        }
        let mask = Tensor::from_slice(&mask, (B, S), &dev)?;

        for strat in [PoolingStrategy::Mean, PoolingStrategy::Cls] {
            let got = pool_on_device(&output, &mask, &strat, B)?;
            let want = cpu_reference(&output, &valid, &strat)?;
            assert!(
                got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-5),
                "on-device pool diverged from CPU reference for {strat:?}",
            );
        }
        Ok(())
    }
}
