use anyhow::Result;

use zti_embed::EmbedEngine;
use zti_rerank::TurboReranker;
use zti_store::chunks_table::ChunkHit;

const KNN_OVERFETCH_MULT: usize = 3;
const DIVERSITY_PENALTY: f32 = 0.04;

pub struct SearchOpts {
    pub limit: usize,
    pub languages: Option<Vec<String>>,
    pub path_glob: Option<String>,
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
    opts: &SearchOpts,
) -> Result<Vec<Hit>> {
    let query_emb = engine.embed_query(query)?;

    let raw_k = opts.limit.saturating_mul(KNN_OVERFETCH_MULT);
    let chunks_table = db.chunks_table(engine.dim()).await?;
    let candidates = chunks_table.knn(&query_emb, raw_k).await?;

    let candidate_indices: Vec<usize> = (0..candidates.len()).collect();
    let ranked = reranker.rerank(&candidate_indices, &query_emb);

    let mut hits: Vec<Hit> = ranked
        .into_iter()
        .filter_map(|(idx, score)| {
            candidates.get(idx).map(|c| Hit {
                chunk: c.clone(),
                score,
            })
        })
        .collect();

    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(opts.limit);

    Ok(hits)
}
