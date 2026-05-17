use zti_protocol::request::ProjectMapReq;
use zti_protocol::response::{ErrorBody, ProjectMapBody, Response};
use zti_tree_sitter::Language;

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::state::DaemonState;

pub async fn handle(req: &ProjectMapReq, state: &DaemonState) -> Response {
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return Response::DslProjectMap(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let index = match ensure_dsl_index(&project, &req.project_root).await {
        Ok(idx) => idx,
        Err(e) => {
            return Response::DslProjectMap(Err(ErrorBody {
                message: e.to_string(),
            }));
        }
    };

    let lang = parse_language(&req.language);
    let file_filter: Option<Vec<u16>> = lang.map(|l| {
        index.files
            .iter()
            .enumerate()
            .filter(|(_, f)| f.language == l)
            .map(|(i, _)| i as u16)
            .collect()
    });

    let max_tokens = req.max_tokens.unwrap_or(8000);
    let renderer = zti_dsl::render::dsl::DslRenderer::new(&index, max_tokens);

    let kind_filter = req.kinds.as_ref().map(|k| parse_kinds(k));

    let text = renderer.render(
        file_filter.as_deref(),
        kind_filter.as_deref(),
    );

    Response::DslProjectMap(Ok(ProjectMapBody { text }))
}

fn parse_language(s: &str) -> Option<Language> {
    match s.to_ascii_lowercase().as_str() {
        "rs" | "rust" => Some(Language::Rust),
        "ts" | "tsx" | "typescript" => Some(Language::Ts),
        "dart" => Some(Language::Dart),
        "sol" | "solidity" => Some(Language::Solidity),
        _ => None,
    }
}

fn parse_kinds(kinds: &[String]) -> Vec<zti_ts_core::types::Kind> {
    kinds.iter().filter_map(|k| zti_ts_core::types::Kind::from_str_lossy(k)).collect()
}
