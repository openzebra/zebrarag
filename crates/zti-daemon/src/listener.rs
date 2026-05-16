use std::sync::Arc;

use anyhow::Result;
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;

use zti_protocol::codec::{read_frame, write_frame};
use zti_protocol::request::Request;
use zti_protocol::response::Response;

use crate::state::DaemonState;

pub async fn run(listener: UnixListener, state: Arc<DaemonState>) -> Result<()> {
    let shutdown = std::pin::pin!(signal::ctrl_c());

    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, _) = accept?;
                let state = state.clone();
                tokio::spawn(handle_connection(stream, state));
            }
            _ = &mut shutdown => {
                tracing::info!("shutting down");
                return Ok(());
            }
        }
    }
}

async fn handle_connection(stream: UnixStream, state: Arc<DaemonState>) {
    let (reader, writer) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(reader);
    let mut writer = tokio::io::BufWriter::new(writer);

    loop {
        let req: Request = match read_frame(&mut reader).await {
            Ok(r) => r,
            Err(_) => return,
        };

        let resp = handle_request(req, &state).await;

        if let Err(e) = write_frame(&mut writer, &resp).await {
            tracing::debug!("write error: {}", e);
            return;
        }

        if matches!(resp, Response::Stop(())) {
            return;
        }
    }
}

async fn handle_request(req: Request, state: &DaemonState) -> Response {
    match req {
        Request::Handshake(h) => crate::handlers::handshake::handle(&h),
        Request::Index(idx) => crate::handlers::index::handle(&idx, state).await,
        Request::Search(s) => crate::handlers::search::handle(&s, state).await,
        Request::ProjectStatus(ps) => crate::handlers::status::handle(&ps, state).await,
        Request::DaemonStatus => crate::handlers::daemon_status::handle(state),
        Request::RemoveProject(rp) => crate::handlers::remove_project::handle(&rp, state).await,
        Request::Stop => Response::Stop(()),
        Request::Doctor(d) => crate::handlers::doctor::handle(&d, state).await,
        Request::DaemonEnv => crate::handlers::env::handle(state),
        Request::DslFileTree(ft) => crate::handlers::dsl_file_tree::handle(&ft, state).await,
        Request::DslProjectMap(pm) => crate::handlers::dsl_project_map::handle(&pm, state).await,
        Request::DslDepTree(dt) => crate::handlers::dsl_dep_tree::handle(&dt, state).await,
        Request::DslSymbolBody(sb) => crate::handlers::dsl_symbol_body::handle(&sb, state).await,
    }
}
