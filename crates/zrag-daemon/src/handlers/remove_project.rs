use zrag_protocol::request::RemoveProjectReq;
use zrag_protocol::response::{ErrorBody, Response};

use crate::state::DaemonState;

pub async fn handle(req: &RemoveProjectReq, state: &DaemonState) -> Response {
    let projects = match zrag_store::list_projects().await {
        Ok(p) => p,
        Err(e) => {
            return Response::RemoveProject(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let row = match projects.iter().find(|p| p.root_path == req.project_root) {
        Some(r) => r,
        None => return Response::RemoveProject(Ok(())),
    };

    let pid: [u8; 32] = match <[u8; 32]>::try_from(row.project_id.as_slice()) {
        Ok(p) => p,
        Err(_) => {
            return Response::RemoveProject(Err(ErrorBody {
                message: "invalid project_id length".into(),
            }));
        }
    };

    state.ann.invalidate(&pid).await;

    {
        let mut reg = state.registry.write().await;
        reg.remove(&pid);
    }

    drop(projects);

    match zrag_common::paths::project_dir_path(&pid) {
        Ok(dir) if dir.exists() => {
            if let Err(e) = std::fs::remove_dir_all(&dir) {
                return Response::RemoveProject(Err(ErrorBody {
                    message: format!("failed to delete project data: {e}"),
                }));
            }
        }
        Err(e) => {
            return Response::RemoveProject(Err(ErrorBody {
                message: format!("failed to resolve project path: {e}"),
            }));
        }
        _ => {}
    }

    if let Some(manager) = state.watch.get() {
        manager.unwatch(&pid).await;
    }

    Response::RemoveProject(Ok(()))
}
