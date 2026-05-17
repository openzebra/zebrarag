use anyhow::Result;

use zti_embed::EmbedEngine;
use zti_rerank::TurboReranker;
use zti_store::chunks_table::ChunkHit;

const KNN_OVERFETCH_MULT: usize = 3;

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
    let query_emb = engine.embed_query_async(query).await?;

    let raw_k = opts.limit.saturating_mul(KNN_OVERFETCH_MULT);
    let chunks_table = db.chunks_table(engine.dim()).await?;
    let candidates = chunks_table
        .knn(
            &query_emb,
            raw_k,
            opts.languages.as_deref(),
            opts.path_glob.as_deref(),
        )
        .await?;

    let rerank_input: Vec<(&[u8], f32)> = candidates
        .iter()
        .map(|c| (c.turbo_code.as_slice(), c.score))
        .collect();

    let ranked = reranker.rerank(&rerank_input, &query_emb);

    let mut hits: Vec<Hit> = ranked
        .into_iter()
        .filter_map(|(idx, score)| {
            candidates.get(idx).map(|c| Hit {
                chunk: c.clone(),
                score,
            })
        })
        .collect();

    hits.truncate(opts.limit);

    Ok(hits)
}
