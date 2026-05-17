use zti_protocol::response::{DaemonStatusInfo, Response};

use crate::state::DaemonState;

pub fn handle(state: &DaemonState) -> Response {
    let registry = state.registry.try_read();
    let projects_loaded = registry.map(|r| r.len()).unwrap_or(0);

    Response::DaemonStatus(DaemonStatusInfo {
        started_at_ns: state.started_at_ns,
        uptime_secs: state.started_at.elapsed().as_secs(),
        projects_loaded,
        model_id: state.engine.profile().model_id.clone(),
        device: state.hardware.device.as_str().to_string(),
    })
}
