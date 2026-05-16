use zti_protocol::request::SymbolBodyReq;
use zti_protocol::response::{ErrorBody, Response, SymbolBodyBody};

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::state::DaemonState;

pub async fn handle(req: &SymbolBodyReq, state: &DaemonState) -> Response {
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return Response::DslSymbolBody(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let index = ensure_dsl_index(&project, &req.project_root).await;

    let sym = match index.symbols.get(req.symbol_id as usize) {
        Some(s) => s,
        None => {
            return Response::DslSymbolBody(Err(ErrorBody {
                message: format!("Symbol {} not found", req.symbol_id),
            }));
        }
    };

    let file = match index.files.get(sym.file_idx as usize) {
        Some(f) => f,
        None => {
            return Response::DslSymbolBody(Err(ErrorBody {
                message: format!("File for symbol {} not found", req.symbol_id),
            }));
        }
    };

    let content = match std::fs::read_to_string(&file.path) {
        Ok(c) => c,
        Err(e) => {
            return Response::DslSymbolBody(Err(ErrorBody {
                message: format!("Failed to read {}: {}", file.path, e),
            }));
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let start = (sym.line as usize).saturating_sub(1);
    let end = (sym.end_line as usize).min(lines.len());

    let body: Vec<&str> = lines[start..end].to_vec();
    let text = format!("// File: {} | Lines: {}-{}\n{}", file.path, sym.line, sym.end_line, body.join("\n"));

    Response::DslSymbolBody(Ok(SymbolBodyBody { text }))
}
