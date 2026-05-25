use zti_protocol::request::RemoveProjectReq;
use zti_protocol::response::Response;

use crate::state::DaemonState;

pub async fn handle(req: &RemoveProjectReq, state: &DaemonState) -> Response {
    let root_path = std::path::Path::new(&req.project_root);
    let pid = zti_common::ids::project_id(root_path);

    state.ann.invalidate(&pid).await;

    {
        let mut reg = state.registry.write().await;
        reg.remove(&pid);
    }

    if let Ok(dir) = zti_common::paths::project_dir_path(&pid) {
        let _ = std::fs::remove_dir_all(&dir);
    }

    Response::RemoveProject(Ok(()))
}
