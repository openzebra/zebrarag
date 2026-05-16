use std::sync::Arc;

use zti_protocol::response::{DaemonStatusInfo, Response};

use crate::state::DaemonState;

pub fn handle(state: &DaemonState) -> Response {
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let registry = state.registry.try_read();
    let projects_loaded = registry.map(|r| r.len()).unwrap_or(0);

    Response::DaemonStatus(DaemonStatusInfo {
        started_at_ns: state.started_at_ns,
        uptime_secs: (now_ns - state.started_at_ns) / 1_000_000_000,
        projects_loaded,
        model_id: state.engine.profile().model_id.clone(),
        device: format!("{:?}", state.hardware.device),
    })
}
