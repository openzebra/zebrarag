use zti_protocol::request::ProjectStatusReq;
use zti_protocol::response::{ErrorBody, ProjectStatus, Response};

use crate::state::DaemonState;

pub async fn handle(req: &ProjectStatusReq, state: &DaemonState) -> Response {
    match &req.project_root {
        Some(root) => {
            let project = match state.load_or_open(root).await {
                Ok(p) => p,
                Err(e) => {
                    return Response::ProjectStatus(Err(ErrorBody {
                        message: e.to_string(),
                    }));
                }
            };

            let dim = state.engine.dim();
            let chunks_len = match project.db.chunks_table(dim).await {
                Ok(t) => t.len().await.unwrap_or(0),
                Err(_) => 0,
            };
            let files_len = match project.db.files_table().await {
                Ok(t) => t.len().await.unwrap_or(0),
                Err(_) => 0,
            };
            let last_indexed_ns = match project.db.projects_table().await {
                Ok(t) => t
                    .get(
                        &zti_common::ids::project_id(std::path::Path::new(root)),
                    )
                    .await
                    .ok()
                    .flatten()
                    .map(|p| p.last_indexed_ns)
                    .unwrap_or(0),
                Err(_) => 0,
            };

            Response::ProjectStatus(Ok(ProjectStatus {
                project_root: root.clone(),
                total_chunks: chunks_len as u64,
                total_files: files_len as u64,
                model_id: state.engine.profile().model_id.clone(),
                model_dim: state.engine.dim() as u32,
                last_indexed_ns,
            }))
        }
        None => {
            let reg = state.registry.read().await;
            let count = reg.len();
            Response::ProjectStatus(Ok(ProjectStatus {
                project_root: format!("{} projects loaded", count),
                total_chunks: 0,
                total_files: 0,
                model_id: state.engine.profile().model_id.clone(),
                model_dim: state.engine.dim() as u32,
                last_indexed_ns: 0,
            }))
        }
    }
}
