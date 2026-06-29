use zrag_protocol::response::{DaemonEnvInfo, Response};

use crate::state::DaemonState;

pub fn handle(state: &DaemonState) -> Response {
    let engine = state.primary_engine();
    let data_dir = zrag_common::paths::data_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let socket_path = zrag_common::paths::daemon_socket()
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    let device = state.device_str().to_owned();

    let (query_prefix, passage_prefix) = match engine.as_ref() {
        zrag_embed::AnyEmbedEngine::Local(local) => (
            local.profile().query_prefix.clone(),
            local.profile().passage_prefix.clone(),
        ),
        zrag_embed::AnyEmbedEngine::Remote(_) => (None, None),
    };

    Response::DaemonEnv(DaemonEnvInfo {
        data_dir,
        socket_path,
        model_id: state.primary_model.to_string(),
        device,
        cpus: state.hardware.cpus as u32,
        mem_total_mb: state.hardware.mem_total / (1024 * 1024),
        model_dim: engine.dim() as u32,
        query_prefix,
        passage_prefix,
    })
}
