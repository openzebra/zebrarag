use zti_protocol::request::SearchReq;
use zti_protocol::response::{ErrorBody, Response, SearchHit, SearchResults};
use zti_rerank::TurboReranker;

use crate::state::DaemonState;

pub async fn handle(req: &SearchReq, state: &DaemonState) -> Response {
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return Response::Search(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let dim = state.engine.dim();
    let chunks_table = match project.db.chunks_table(dim).await {
        Ok(t) => t,
        Err(e) => {
            return Response::Search(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let query_emb = match state.engine.embed_query_async(&req.query).await {
        Ok(e) => e,
        Err(e) => {
            return Response::Search(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let k = req.limit * 3;
    let candidates = match chunks_table
        .knn(
            &query_emb,
            k,
            req.languages.as_deref(),
            req.path_glob.as_deref(),
        )
        .await
    {
        Ok(c) => c,
        Err(e) => {
            return Response::Search(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let reranker = match TurboReranker::new(dim) {
        Ok(r) => r,
        Err(e) => {
            return Response::Search(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let rerank_input: Vec<(&[u8], f32)> = candidates
        .iter()
        .map(|c| (c.turbo_code.as_slice(), c.score))
        .collect();
    let ranked = reranker.rerank(&rerank_input, &query_emb);

    let hits: Vec<SearchHit> = ranked
        .into_iter()
        .take(req.limit)
        .filter_map(|(idx, score)| {
            candidates.get(idx).map(|c| SearchHit {
                chunk_id: c.chunk_id.clone(),
                file_path: c.file_path.clone(),
                symbol_qualified: c.symbol_qualified.clone(),
                start_line: c.start_line,
                end_line: c.end_line,
                content: c.content.clone(),
                score,
            })
        })
        .collect();

    let total = hits.len();
    Response::Search(Ok(SearchResults { hits, total }))
}
