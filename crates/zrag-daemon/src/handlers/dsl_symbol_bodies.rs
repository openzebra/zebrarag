use zrag_protocol::request::SymbolBodiesReq;
use zrag_protocol::response::{Response, SymbolBodiesBody};

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::handlers::with_project;
use crate::state::DaemonState;

pub async fn handle(req: &SymbolBodiesReq, state: &DaemonState) -> Response {
    let project_root = req.project_root.clone();
    let symbol_ids = req.symbol_ids.clone();

    let result = with_project(state, &req.project_root, |project| async move {
        let index = ensure_dsl_index(&project, &project_root).await?;
        let entries = zrag_dsl::resolve_symbol_bodies(&index, &symbol_ids);
        Ok(SymbolBodiesBody { entries })
    })
    .await;

    Response::DslSymbolBodies(result)
}
