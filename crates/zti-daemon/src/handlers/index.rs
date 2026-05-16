use std::sync::Arc;

use zti_protocol::request::IndexReq;
use zti_protocol::response::{ErrorBody, IndexStats, Response};

use crate::state::DaemonState;

pub async fn handle(req: &IndexReq, state: &DaemonState) -> Response {
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return Response::Index(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let _lock = project.indexing_lock.lock().await;

    let root = std::path::Path::new(&req.project_root);
    let engine = state.engine.clone();
    let db = project.db.clone();

    match zti_pipeline::indexer::index_project(
        root,
        &engine,
        &db,
        &zti_pipeline::progress::SilentReporter,
    )
    .await
    {
        Ok(stats) => Response::Index(Ok(IndexStats {
            total_chunks: stats.total_chunks,
            total_files: stats.total_files,
            new_chunks: stats.new_chunks,
            reindexed_files: stats.reindexed_files,
            duration_ms: stats.duration_ms,
        })),
        Err(e) => Response::Index(Err(ErrorBody {
            message: e.to_string(),
        })),
    }
}
