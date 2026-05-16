use zti_protocol::request::ProjectMapReq;
use zti_protocol::response::{ErrorBody, ProjectMapBody, Response};

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

    let index = ensure_dsl_index(&project, &req.project_root).await;

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

fn parse_language(s: &str) -> Option<zti_dsl::model::Language> {
    match s.to_ascii_lowercase().as_str() {
        "rs" | "rust" => Some(zti_dsl::model::Language::Rust),
        "ts" | "tsx" | "typescript" => Some(zti_dsl::model::Language::TypeScript),
        "dart" => Some(zti_dsl::model::Language::Dart),
        "sol" | "solidity" => Some(zti_dsl::model::Language::Solidity),
        _ => None,
    }
}

fn parse_kinds(kinds: &[String]) -> Vec<zti_dsl::model::Kind> {
    kinds.iter().filter_map(|k| match k.as_str() {
        "fn" | "function" => Some(zti_dsl::model::Kind::Function),
        "method" => Some(zti_dsl::model::Kind::Method),
        "struct" => Some(zti_dsl::model::Kind::Struct),
        "enum" => Some(zti_dsl::model::Kind::Enum),
        "class" => Some(zti_dsl::model::Kind::Class),
        "interface" => Some(zti_dsl::model::Kind::Interface),
        "const" => Some(zti_dsl::model::Kind::Const),
        "static" => Some(zti_dsl::model::Kind::Static),
        "module" | "mod" => Some(zti_dsl::model::Kind::Module),
        _ => None,
    }).collect()
}
