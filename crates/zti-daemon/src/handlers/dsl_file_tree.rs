use std::sync::Arc;

use zti_protocol::request::FileTreeReq;
use zti_protocol::response::{ErrorBody, FileTreeBody, Response};

use crate::state::DaemonState;

pub async fn handle(req: &FileTreeReq, state: &DaemonState) -> Response {
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return Response::DslFileTree(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let index = match ensure_dsl_index(&project, &req.project_root).await {
        Ok(idx) => idx,
        Err(e) => {
            return Response::DslFileTree(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let file_indices: Vec<u16> = (0..index.files.len() as u16).collect();
    let text = zti_dsl::render::dsl::render_files_only(&index, &file_indices);

    Response::DslFileTree(Ok(FileTreeBody { text }))
}

pub async fn ensure_dsl_index(
    project: &crate::state::LoadedProject,
    root: &str,
) -> anyhow::Result<Arc<zti_dsl::ProjectIndex>> {
    {
        let guard = project.dsl_index.read().await;
        if let Some(ref idx) = *guard {
            return Ok(Arc::clone(idx));
        }
    }

    let idx = Arc::new(zti_dsl::build_index(root)?);
    {
        let mut guard = project.dsl_index.write().await;
        if guard.is_none() {
            *guard = Some(Arc::clone(&idx));
        }
    }
    Ok(idx)
}
