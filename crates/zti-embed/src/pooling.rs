pub enum PoolingStrategy {
    Mean,
    Cls,
}

/// Pool a single row from a contiguous (seq * dim) slice into a `Vec<f32>` of
/// length `dim`. Mean pooling respects the attention mask; CLS takes index 0.
/// Zero-copy on the input; returns the pooled vector owned (downstream
/// normalize mutates it in-place, so we can't return a borrow).
pub fn pool_row(strategy: &PoolingStrategy, data: &[f32], dim: usize, valid: usize) -> Vec<f32> {
    match strategy {
        PoolingStrategy::Mean => {
            let mut sum = vec![0.0f32; dim];
            if valid == 0 {
                return sum;
            }
            for j in 0..valid {
                let row = &data[j * dim..(j + 1) * dim];
                for (s, &v) in sum.iter_mut().zip(row) {
                    *s += v;
                }
            }
            let c = valid as f32;
            for x in &mut sum {
                *x /= c;
            }
            sum
        }
        PoolingStrategy::Cls => data[..dim].to_vec(),
    }
}
