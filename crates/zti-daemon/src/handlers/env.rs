use zti_protocol::response::{DaemonEnvInfo, Response};

use crate::state::DaemonState;

pub fn handle(state: &DaemonState) -> Response {
    let data_dir = zti_common::paths::data_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let socket_path = zti_common::paths::daemon_socket()
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    Response::DaemonEnv(DaemonEnvInfo {
        data_dir,
        socket_path,
        model_id: state.engine.profile().model_id.clone(),
        device: format!("{:?}", state.hardware.device),
        cpus: state.hardware.cpus,
        mem_total_mb: state.hardware.mem_total / (1024 * 1024),
    })
}
