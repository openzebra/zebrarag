use std::sync::atomic::Ordering;

use zti_protocol::request::CancelIndexReq;
use zti_protocol::response::Response;

use crate::state::DaemonState;

pub async fn handle(req: &CancelIndexReq, state: &DaemonState) -> Response {
    let root = match std::path::Path::new(&req.project_root).canonicalize() {
        Ok(r) => r,
        Err(_) => return Response::CancelIndex(Ok(())),
    };
    let pid = zti_common::ids::project_id(&root);

    let reg = state.registry.read().await;
    match reg.get(&pid) {
        Some(project) => {
            project.cancel.store(true, Ordering::Relaxed);
            Response::CancelIndex(Ok(()))
        }
        None => Response::CancelIndex(Ok(())),
    }
}
