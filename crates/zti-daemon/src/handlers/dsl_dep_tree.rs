use zti_protocol::request::DepTreeReq;
use zti_protocol::response::{DepTreeBody, ErrorBody, Response};

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::state::DaemonState;

pub async fn handle(req: &DepTreeReq, state: &DaemonState) -> Response {
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return Response::DslDepTree(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let index = ensure_dsl_index(&project, &req.project_root).await;

    let depth = req.depth.unwrap_or(2);
    let renderer = zti_dsl::AsciiTreeRenderer::new(&index);

    let text = match req.direction.as_str() {
        "callers" => renderer.render_callers(req.symbol_id, depth),
        "callees" => renderer.render_callees(req.symbol_id, depth),
        _ => "Error: direction must be 'callers' or 'callees'".to_string(),
    };

    Response::DslDepTree(Ok(DepTreeBody { text }))
}
