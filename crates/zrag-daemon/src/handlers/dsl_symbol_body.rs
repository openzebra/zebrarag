use zrag_common::dsl::SymbolBodyEntry;
use zrag_protocol::request::SymbolBodyReq;
use zrag_protocol::response::{Response, SymbolBodyBody};

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::handlers::with_project;
use crate::state::DaemonState;

pub async fn handle(req: &SymbolBodyReq, state: &DaemonState) -> Response {
    let project_root = req.project_root.clone();
    let symbol_id = req.symbol_id;

    let result = with_project(state, &req.project_root, |project| async move {
        let index = ensure_dsl_index(&project, &project_root).await?;
        let entries = zrag_dsl::resolve_symbol_bodies(&index, &[symbol_id]);
        match entries.into_iter().next() {
            Some(SymbolBodyEntry::Ok {
                kind_short,
                symbol_id,
                start_line,
                end_line,
                body,
                ..
            }) => {
                let text = format!(
                    "{}#{} : {}-{}\n{}",
                    kind_short, symbol_id, start_line, end_line, body
                );
                Ok(SymbolBodyBody { text })
            }
            Some(SymbolBodyEntry::Err { message, .. }) => Err(anyhow::anyhow!(message)),
            None => Err(anyhow::anyhow!("Symbol {} not found", symbol_id)),
        }
    })
    .await;

    Response::DslSymbolBody(result)
}
