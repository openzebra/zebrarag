use zrag_protocol::request::DepTreeReq;
use zrag_protocol::response::{DepTreeBody, Response};

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::handlers::with_project;
use crate::state::DaemonState;

pub async fn handle(req: &DepTreeReq, state: &DaemonState) -> Response {
    let project_root = req.project_root.clone();
    let direction = req.direction.clone();
    let symbol_id = req.symbol_id;
    let depth = req.depth.unwrap_or(2);

    let result = with_project(state, &req.project_root, |project| async move {
        let index = ensure_dsl_index(&project, &project_root).await?;
        let renderer = zrag_dsl::AsciiTreeRenderer::new(&index);
        let text = match direction.as_str() {
            "callers" => renderer.render_callers(symbol_id, depth),
            "callees" => renderer.render_callees(symbol_id, depth, false),
            other => {
                anyhow::bail!("direction must be 'callers' or 'callees', got '{}'", other);
            }
        };
        Ok(DepTreeBody { text })
    })
    .await;

    Response::DslDepTree(result)
}
