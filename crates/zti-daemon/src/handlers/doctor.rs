use zti_protocol::request::DoctorReq;
use zti_protocol::response::{DoctorReport, Response};

use crate::state::DaemonState;

pub async fn handle(req: &DoctorReq, state: &DaemonState) -> Response {
    let model_path = state.engine.profile().onnx_path.display().to_string();
    let model_ok = std::path::Path::new(&model_path).exists();

    let model_probe = if model_ok {
        match state.engine.embed_batch_async(&["hello"]).await {
            Ok(embs) => {
                let dim_ok = embs.first().map(|e| e.len() == state.engine.dim()).unwrap_or(false);
                dim_ok
            }
            Err(_) => false,
        }
    } else {
        false
    };

    let (db_ok, db_path, chunk_count) = match &req.project_root {
        Some(root) => {
            let pid = zti_common::ids::project_id(std::path::Path::new(root));
            let db_path = zti_common::paths::project_dir(&pid)
                .map(|p| p.join("lance").display().to_string())
                .unwrap_or_default();
            let path_exists = std::path::Path::new(&db_path).exists();

            let count = if path_exists {
                match state.load_or_open(root).await {
                    Ok(proj) => {
                        let table = proj.db.chunks_table(state.engine.dim()).await;
                        match table {
                            Ok(t) => t.len().await.unwrap_or(0),
                            Err(_) => 0,
                        }
                    }
                    Err(_) => 0,
                }
            } else {
                0
            };

            (path_exists && count > 0, db_path, count)
        }
        None => {
            let db_path = zti_common::paths::data_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            (std::path::Path::new(&db_path).exists(), db_path, 0)
        }
    };

    Response::Doctor(Ok(DoctorReport {
        model_ok: model_ok && model_probe,
        model_path,
        db_ok,
        db_path,
        device: state.hardware.device.as_str().to_string(),
    }))
}
