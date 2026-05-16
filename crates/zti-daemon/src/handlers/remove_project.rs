use zti_protocol::request::RemoveProjectReq;
use zti_protocol::response::{ErrorBody, Response};

use crate::state::DaemonState;

pub async fn handle(req: &RemoveProjectReq, state: &DaemonState) -> Response {
    let root_path = std::path::Path::new(&req.project_root);
    let pid = zti_common::ids::project_id(root_path);

    let mut reg = state.registry.write().await;
    reg.remove(&pid);

    Response::RemoveProject(Ok(()))
}
