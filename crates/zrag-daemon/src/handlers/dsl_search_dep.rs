use zrag_protocol::request::SearchDepReq;
use zrag_protocol::response::{Response, SearchDepBody};

use crate::handlers::dsl_file_tree::ensure_dsl_index;
use crate::handlers::with_project;
use crate::state::DaemonState;

pub async fn handle(req: &SearchDepReq, state: &DaemonState) -> Response {
    let max_tokens = req.max_tokens.unwrap_or(6000);
    let depth = req.depth.unwrap_or(2);
    let result = with_project(state, &req.project_root, |project| {
        let name = req.name.clone();
        let root = req.project_root.clone();
        async move {
            let index = ensure_dsl_index(&project, &root).await?;
            let text = match zrag_dsl::resolve_name(&index, &name) {
                zrag_dsl::NameMatch::Found(id) => {
                    zrag_dsl::render_symbol_overview(&index, id, depth, max_tokens)
                }
                zrag_dsl::NameMatch::Ambiguous(ids) => {
                    zrag_dsl::search_dep::render_candidates(&index, &ids)
                }
                zrag_dsl::NameMatch::NotFound => format!(
                    "No symbol named '{}'. Try projectList / fileTree, or index its \
                     source as a project first.",
                    name
                ),
            };
            Ok(SearchDepBody { text })
        }
    })
    .await;

    Response::DslSearchDep(result)
}
