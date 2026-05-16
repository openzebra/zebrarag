use anyhow::Result;

pub enum PoolingStrategy {
    Mean,
    Cls,
}

pub fn pool(strategy: &PoolingStrategy, embeddings: &[&[f32]], attention_mask: &[u32]) -> Result<Vec<f32>> {
    if embeddings.is_empty() {
        anyhow::bail!("empty embeddings for pooling");
    }
    let dim = embeddings[0].len();
    match strategy {
        PoolingStrategy::Mean => {
            let mut sum = vec![0.0f32; dim];
            let mut count = 0u32;
            for (i, emb) in embeddings.iter().enumerate() {
                if attention_mask.get(i).copied().unwrap_or(1) == 1 {
                    for (j, s) in sum.iter_mut().enumerate() {
                        *s += emb[j];
                    }
                    count += 1;
                }
            }
            if count == 0 {
                count = 1;
            }
            Ok(sum.into_iter().map(|v| v / count as f32).collect())
        }
        PoolingStrategy::Cls => {
            let cls = embeddings.first().ok_or_else(|| anyhow::anyhow!("no CLS token"))?;
            Ok(cls.to_vec())
        }
    }
}
