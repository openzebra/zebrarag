use zti_protocol::request::ProjectStatusReq;
use zti_protocol::response::{ErrorBody, ProjectStatus, Response};

use crate::state::DaemonState;

pub async fn handle(req: &ProjectStatusReq, state: &DaemonState) -> Response {
    match &req.project_root {
        Some(root) => {
            let root_path = std::path::Path::new(root);
            let pid = zti_common::ids::project_id(root_path);

            let reg = state.registry.read().await;
            match reg.get(&pid) {
                Some(_project) => Response::ProjectStatus(Ok(ProjectStatus {
                    project_root: root.clone(),
                    total_chunks: 0,
                    total_files: 0,
                    model_id: state.engine.profile().model_id.clone(),
                    model_dim: state.engine.dim() as u32,
                    last_indexed_ns: 0,
                })),
                None => Response::ProjectStatus(Err(ErrorBody {
                    message: format!("project not loaded: {}", root),
                })),
            }
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
