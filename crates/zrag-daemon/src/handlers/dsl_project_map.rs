use zrag_protocol::request::ProjectMapReq;
use zrag_protocol::response::{ProjectMapBody, Response};
use zrag_tree_sitter::{parse_kinds, parse_language};

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::handlers::with_project;
use crate::state::DaemonState;

pub async fn handle(req: &ProjectMapReq, state: &DaemonState) -> Response {
    let project_root = req.project_root.clone();
    let kinds = req.kinds.clone();
    let max_tokens = req.max_tokens.unwrap_or(8000);

    let result = with_project(state, &req.project_root, |project| async move {
        let index = ensure_dsl_index(&project, &project_root).await?;

        let file_filter: Option<Vec<u16>> = req.language.as_ref().and_then(|l| {
            let lang = parse_language(l)?;
            Some(
                index
                    .files
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.language == lang)
                    .map(|(i, _)| i as u16)
                    .collect(),
            )
        });

        let kind_filter = kinds.as_ref().map(|k| parse_kinds(k));

        let renderer = zrag_dsl::render::dsl::DslRenderer::new(&index, max_tokens);
        let text = renderer.render(file_filter.as_deref(), kind_filter.as_deref());

        Ok(ProjectMapBody { text })
    })
    .await;

    Response::DslProjectMap(result)
}
