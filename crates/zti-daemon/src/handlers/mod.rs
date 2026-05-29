use std::future::Future;
use std::sync::Arc;

use zti_protocol::response::ErrorBody;

use crate::state::{DaemonState, LoadedProject};

pub mod cancel_index;
pub mod daemon_status;
pub mod doctor;
pub mod dsl_dep_tree;
pub mod dsl_file_tree;
pub mod dsl_project_map;
pub mod dsl_symbol_bodies;
pub mod dsl_symbol_body;
pub mod env;
pub mod handshake;
pub mod index;
pub mod remove_project;
pub mod search;
pub mod status;

/// Single source of truth for the `load_or_open(..) -> match -> ErrorBody`
/// pattern that every project-scoped handler used to duplicate. The handler
/// supplies the body closure; this helper handles both the open-failure and
/// body-failure paths, mapping `anyhow::Error` to `ErrorBody` once.
pub async fn with_project<F, Fut, T>(
    state: &DaemonState,
    project_root: &str,
    body: F,
) -> Result<T, ErrorBody>
where
    F: FnOnce(Arc<LoadedProject>) -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    let project = state
        .load_or_open(project_root)
        .await
        .map_err(|e| ErrorBody {
            message: e.to_string(),
        })?;
    body(project).await.map_err(|e| ErrorBody {
        message: e.to_string(),
    })
}
