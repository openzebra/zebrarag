use std::sync::Arc;

use anyhow::Result;
use tokio::io::{BufReader, BufWriter};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal;

use zti_protocol::codec::{read_frame, write_frame};
use zti_protocol::request::Request;
use zti_protocol::response::Response;

use crate::state::DaemonState;

pub async fn run(listener: UnixListener, state: Arc<DaemonState>) -> Result<()> {
    let mut shutdown_rx = state.shutdown_rx.clone();

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (stream, _) = accept?;
                let state = state.clone();
                let mut shutdown_rx_clone = state.shutdown_rx.clone();
                tokio::spawn(async move {
                    handle_connection(stream, state, &mut shutdown_rx_clone).await;
                });
            }
            _ = signal::ctrl_c() => {
                tracing::info!("ctrl-c received, shutting down");
                let _ = state.shutdown_tx.send(true);
                break;
            }
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("shutdown signal received");
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn handle_connection(
    stream: UnixStream,
    state: Arc<DaemonState>,
    shutdown_rx: &mut tokio::sync::watch::Receiver<bool>,
) {
    let (reader, writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut writer = BufWriter::new(writer);

    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        let req: Request = match read_frame(&mut reader).await {
            Ok(r) => r,
            Err(_) => return,
        };

        // Index streams its own frames. Everything else returns a single
        // Response that we write here.
        let stop = match req {
            Request::Index(idx) => {
                if let Err(e) =
                    crate::handlers::index::handle_streaming(&mut writer, &idx, &state).await
                {
                    tracing::debug!("index streaming error: {}", e);
                    return;
                }
                false
            }
            other => {
                let resp = handle_request(other, &state).await;
                let is_stop = matches!(resp, Response::Stop(()));
                if let Err(e) = write_frame(&mut writer, &resp).await {
                    tracing::debug!("write error: {}", e);
                    return;
                }
                if is_stop {
                    let _ = state.shutdown_tx.send(true);
                }
                is_stop
            }
        };

        if stop {
            return;
        }
    }
}

async fn handle_request(req: Request, state: &DaemonState) -> Response {
    match req {
        Request::Handshake(h) => crate::handlers::handshake::handle(&h),
        Request::Index(_) => unreachable!("Index is handled in streaming path"),
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

