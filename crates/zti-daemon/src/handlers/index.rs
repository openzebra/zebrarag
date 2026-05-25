use anyhow::Result;
use tokio::io::AsyncWrite;
use tokio::sync::mpsc;

use zti_pipeline::progress::IpcReporter;
use zti_protocol::codec::write_frame;
use zti_protocol::request::IndexReq;
use zti_protocol::response::{ErrorBody, IndexStats, Response};

use crate::state::DaemonState;

/// Streaming index. Writes any number of `Response::IndexProgress` frames
/// followed by exactly one terminal `Response::Index(Ok|Err)`.
pub async fn handle_streaming<W>(writer: &mut W, req: &IndexReq, state: &DaemonState) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let project = match state.load_or_open(&req.project_root).await {
        Ok(p) => p,
        Err(e) => {
            return write_frame(
                writer,
                &Response::Index(Err(ErrorBody {
                    message: e.to_string(),
                })),
            )
            .await;
        }
    };

    let _lock = project.indexing_lock.lock().await;

    let root = std::path::Path::new(&req.project_root).to_path_buf();
    let engine = state.primary_engine();
    let db = project.db.clone();

    let (tx, mut rx) = mpsc::unbounded_channel();
    let reporter = IpcReporter::new(tx);

    let mut indexing = Box::pin(async move {
        zti_pipeline::indexer::index_project(&root, &engine, &db, &reporter).await
    });

    let final_result = loop {
        tokio::select! {
            biased;
            Some(p) = rx.recv() => {
                if let Err(e) = write_frame(writer, &Response::IndexProgress(p)).await {
                    tracing::debug!("progress write error: {}", e);
                    return Err(e);
                }
            }
            r = &mut indexing => {
                break r;
            }
        }
    };

    // Drain any progress events that landed after indexing returned.
    while let Ok(p) = rx.try_recv() {
        let _ = write_frame(writer, &Response::IndexProgress(p)).await;
    }

    let terminal = match final_result {
        Ok(stats) => Response::Index(Ok(IndexStats {
            total_chunks: stats.total_chunks,
            total_files: stats.total_files,
            new_chunks: stats.new_chunks,
            reindexed_files: stats.reindexed_files,
            duration_ms: stats.duration_ms,
        })),
        Err(e) => Response::Index(Err(ErrorBody {
            message: e.to_string(),
        })),
    };
    write_frame(writer, &terminal).await
}
