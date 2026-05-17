pub enum PoolingStrategy {
    Mean,
    Cls,
}

/// Pool a single row from a contiguous (seq * dim) slice into a `Vec<f32>` of
/// length `dim`. Mean pooling respects the attention mask; CLS takes index 0.
/// Zero-copy on the input; returns the pooled vector owned (downstream
/// normalize mutates it in-place, so we can't return a borrow).
pub fn pool_row(
    strategy: &PoolingStrategy,
    data: &[f32],
    dim: usize,
    seq: usize,
    mask: &[u32],
) -> Vec<f32> {
    match strategy {
        PoolingStrategy::Mean => {
            let mut sum = vec![0.0f32; dim];
            let mut count = 0u32;
            for j in 0..seq {
                if mask.get(j).copied().unwrap_or(1) == 1 {
                    let off = j * dim;
                    for k in 0..dim {
                        sum[k] += data[off + k];
                    }
                    count += 1;
                }
            }
            let c = if count == 0 { 1.0 } else { count as f32 };
            for x in &mut sum {
                *x /= c;
            }
            sum
        }
        PoolingStrategy::Cls => data[..dim].to_vec(),
    }
}
