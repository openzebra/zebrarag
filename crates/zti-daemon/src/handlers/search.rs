use std::borrow::Cow;
use std::collections::HashMap;

use zti_dsl::LEGEND_LINE;
use zti_protocol::request::SearchReq;
use zti_protocol::response::{Response, SearchHit, SearchResults};
use zti_rerank::TurboReranker;
use zti_store::chunks_table::ChunkHit;

use crate::handlers::with_project;
use crate::state::DaemonState;

const DIVERSITY_PENALTY: f32 = 0.04;
const APPENDIX_CAP: usize = 8;

pub async fn handle(req: &SearchReq, state: &DaemonState) -> Response {
    let query = req.query.clone();
    let limit = req.limit;
    let languages = req.languages.clone();
    let path_glob = req.path_glob.clone();

    let result = with_project(state, &req.project_root, |project| async move {
        let dim = state.engine.dim();
        let chunks_table = project.db.chunks_table(dim).await?;
        let query_emb = state.engine.embed_query_async(&query).await?;

        let k = limit.saturating_mul(3);
        let candidates = chunks_table
            .knn(&query_emb, k, languages.as_deref(), path_glob.as_deref())
            .await?;

        let reranker = TurboReranker::new(dim)?;
        let rerank_input: Vec<(&[u8], f32)> = candidates
            .iter()
            .map(|c| (c.turbo_code.as_slice(), c.score))
            .collect();
        let ranked = reranker.rerank(&rerank_input, &query_emb);

        // Diversify: penalize repeated parents so siblings don't crowd out
        // the top-K. Borrows ChunkHit refs — no clone.
        let seeded: Vec<(&ChunkHit, f32)> = ranked
            .into_iter()
            .filter_map(|(idx, score)| candidates.get(idx).map(|c| (c, score)))
            .collect();

        let mut parents_seen: HashMap<u32, usize> =
            HashMap::with_capacity(seeded.len());
        let mut diversified: Vec<(&ChunkHit, f32)> = seeded
            .into_iter()
            .map(|(c, score)| {
                let Some(parent) = c.parent_sym_id else {
                    return (c, score);
                };
                let count = parents_seen.entry(parent).or_insert(0);
                let adjusted = score - (*count as f32) * DIVERSITY_PENALTY;
                *count += 1;
                (c, adjusted)
            })
            .collect();
        diversified.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        diversified.truncate(limit);

        let hits: Vec<SearchHit> = diversified
            .iter()
            .map(|(c, score)| chunk_to_hit(c, *score, &req.project_root))
            .collect();

        // Appendix: union of per-seed precomputed callees, minus seed
        // sym_ids, capped at APPENDIX_CAP. One IN-list lookup, no BFS.
        let mut seen: std::collections::HashSet<u32> =
            std::collections::HashSet::with_capacity(hits.len() + APPENDIX_CAP);
        for h in &hits {
            seen.insert(h.sym_id);
        }
        let mut appendix_ids: Vec<u32> = Vec::with_capacity(APPENDIX_CAP);
        for (c, _) in &diversified {
            for &sid in &c.appendix_sym_ids {
                if appendix_ids.len() >= APPENDIX_CAP {
                    break;
                }
                if seen.insert(sid) {
                    appendix_ids.push(sid);
                }
            }
            if appendix_ids.len() >= APPENDIX_CAP {
                break;
            }
        }

        let appendix = if appendix_ids.is_empty() {
            Vec::new()
        } else {
            let rows = chunks_table.get_by_sym_ids(&appendix_ids).await?;
            // Preserve the appendix_ids ordering (BFS-ish across seeds).
            let by_sym: HashMap<u32, &ChunkHit> =
                rows.iter().map(|r| (r.sym_id, r)).collect();
            let mut out: Vec<SearchHit> = Vec::with_capacity(appendix_ids.len());
            for sid in &appendix_ids {
                if let Some(c) = by_sym.get(sid) {
                    out.push(chunk_to_hit(c, 0.0, &req.project_root));
                }
            }
            out
        };

        let total = hits.len();

        Ok(SearchResults {
            hits,
            appendix,
            legend: Cow::Borrowed(LEGEND_LINE),
            total,
        })
    })
    .await;

    Response::Search(result)
}

fn chunk_to_hit(c: &ChunkHit, score: f32, project_root: &str) -> SearchHit {
    let rel = c
        .file_path
        .strip_prefix(project_root)
        .unwrap_or(&c.file_path)
        .trim_start_matches('/');
    SearchHit {
        chunk_id: c.chunk_id.clone(),
        file_path: rel.to_string(),
        symbol_qualified: c.symbol_qualified.clone(),
        symbol_kind: c.symbol_kind.clone(),
        sym_id: c.sym_id,
        start_line: c.start_line,
        end_line: c.end_line,
        content: c.content.clone(),
        score,
    }
}
