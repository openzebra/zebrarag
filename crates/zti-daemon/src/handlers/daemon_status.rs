use zti_protocol::response::{DaemonStatusInfo, Response};

use crate::state::DaemonState;

pub async fn handle(state: &DaemonState) -> Response {
    let registry = state.registry.try_read();
    let projects_loaded = registry.map(|r| r.len()).unwrap_or(0);

    let loaded_models = {
        let engines = state.engines.read().await;
        engines.keys().map(|k| k.to_string()).collect()
    };
    let loading_model = state.loading_model.read().await.as_deref().map(str::to_string);

    let engine = state.primary_engine();
    let device = engine.hardware().device.as_str().to_owned();

    Response::DaemonStatus(DaemonStatusInfo {
        started_at_ns: state.started_at_ns,
        uptime_secs: state.started_at.elapsed().as_secs(),
        projects_loaded,
        model_id: state.primary_model.to_string(),
        device,
        loaded_models,
        loading_model,
    })
}
