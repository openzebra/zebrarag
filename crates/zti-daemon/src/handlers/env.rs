use zti_protocol::response::{DaemonEnvInfo, Response};

use crate::state::DaemonState;

pub fn handle(state: &DaemonState) -> Response {
    let engine = state.primary_engine();
    let data_dir = zti_common::paths::data_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let socket_path = zti_common::paths::daemon_socket()
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    let ep = engine.hardware().ep_status.get();
    let device = ep.device_label(&state.hardware.device).into_owned();

    Response::DaemonEnv(DaemonEnvInfo {
        data_dir,
        socket_path,
        model_id: state.primary_model.to_string(),
        device,
        cpus: state.hardware.cpus as u32,
        mem_total_mb: state.hardware.mem_total / (1024 * 1024),
        query_prefix: engine.profile().query_prefix.clone(),
        passage_prefix: engine.profile().passage_prefix.clone(),
    })
}
