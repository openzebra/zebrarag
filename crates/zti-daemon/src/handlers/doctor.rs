use zti_protocol::request::DoctorReq;
use zti_protocol::response::{DoctorReport, ErrorBody, Response};

use crate::state::DaemonState;

pub async fn handle(req: &DoctorReq, state: &DaemonState) -> Response {
    let model_path = state.engine.profile().onnx_path.display().to_string();
    let model_ok = std::path::Path::new(&model_path).exists();

    let db_path = match &req.project_root {
        Some(root) => {
            let pid = zti_common::ids::project_id(std::path::Path::new(root));
            zti_common::paths::project_dir(&pid)
                .map(|p| p.join("lance").display().to_string())
                .unwrap_or_default()
        }
        None => zti_common::paths::data_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
    };
    let db_ok = std::path::Path::new(&db_path).exists();

    Response::Doctor(Ok(DoctorReport {
        model_ok,
        model_path,
        db_ok,
        db_path,
        device: format!("{:?}", state.hardware.device),
    }))
}
