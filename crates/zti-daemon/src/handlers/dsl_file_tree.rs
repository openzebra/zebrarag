use std::sync::Arc;

use zti_protocol::request::FileTreeReq;
use zti_protocol::response::{FileTreeBody, Response};

use crate::handlers::with_project;
use crate::state::{DaemonState, LoadedProject};

pub async fn handle(req: &FileTreeReq, state: &DaemonState) -> Response {
    let project_root = req.project_root.clone();
    let result = with_project(state, &req.project_root, |project| async move {
        let index = ensure_dsl_index(&project, &project_root).await?;
        let file_indices: Vec<u16> = (0..index.files.len() as u16).collect();
        Ok(FileTreeBody {
            text: zti_dsl::render::dsl::render_files_only(&index, &file_indices),
        })
    })
    .await;
    Response::DslFileTree(result)
}

pub async fn ensure_dsl_index(
    project: &LoadedProject,
    root: &str,
) -> anyhow::Result<Arc<zti_dsl::ProjectIndex>> {
    {
        let guard = project.dsl_index.read().await;
        if let Some(ref idx) = *guard {
            return Ok(Arc::clone(idx));
        }
    }

    let (idx_inner, _text_files) = zti_dsl::build_index(root)?;
    let idx = Arc::new(idx_inner);
    {
        let mut guard = project.dsl_index.write().await;
        if guard.is_none() {
            *guard = Some(Arc::clone(&idx));
        }
    }
    Ok(idx)
}
