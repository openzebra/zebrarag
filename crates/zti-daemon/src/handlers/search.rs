use std::sync::Arc;

use zti_protocol::request::SearchReq;
use zti_protocol::response::{ErrorBody, Response, SearchHit, SearchResults};

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

    let query_emb = match state.engine.embed_query(&req.query) {
        Ok(e) => e,
        Err(e) => {
            return Response::Search(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let k = req.limit * 3;
    let candidates = match chunks_table.knn(&query_emb, k).await {
        Ok(c) => c,
        Err(e) => {
            return Response::Search(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let hits: Vec<SearchHit> = candidates
        .into_iter()
        .take(req.limit)
        .map(|c| SearchHit {
            chunk_id: Vec::new(),
            file_path: c.file_path,
            symbol_qualified: c.symbol_qualified,
            start_line: c.start_line,
            end_line: c.end_line,
            content: c.content,
            score: c.score,
        })
        .collect();

    let total = hits.len();
    Response::Search(Ok(SearchResults { hits, total }))
}
