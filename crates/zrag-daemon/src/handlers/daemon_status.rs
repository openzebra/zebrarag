use zrag_protocol::response::{DaemonStatusInfo, Response};

use crate::state::DaemonState;

pub async fn handle(state: &DaemonState) -> Response {
    let registry = state.registry.try_read();
    let projects_loaded = registry.map(|r| r.len()).unwrap_or(0);

    let loaded_models = {
        let engines = state.engines.read().await;
        engines.keys().map(|k| k.to_string()).collect()
    };
    let device = state.device_str().to_owned();

    Response::DaemonStatus(DaemonStatusInfo {
        started_at_ns: state.started_at_ns,
        uptime_secs: state.started_at.elapsed().as_secs(),
        projects_loaded,
        model_id: state.primary_model.to_string(),
        device,
        cpus: state.hardware.cpus as u32,
        mem_total_mb: state.hardware.mem_total / (1024 * 1024),
        model_dtype: state.model_dtype.clone(),
        loaded_models,
        loading_model: None,
    })
}
