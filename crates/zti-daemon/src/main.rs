use std::sync::Arc;

use anyhow::Result;
use tokio::net::UnixListener;
use tracing_subscriber::EnvFilter;

mod handlers;
mod listener;
mod registry;
mod state;

use state::DaemonState;

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let log_path = zti_common::paths::daemon_log()?;
    let log_file = std::fs::File::create(&log_path)?;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(log_file)
        .with_ansi(false)
        .init();

    let pid_path = zti_common::paths::daemon_pid()?;
    let pid = std::process::id().to_string();
    std::fs::write(&pid_path, &pid)?;

    let socket_path = zti_common::paths::daemon_socket()?;
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    let model_id = std::env::var("ZEBRA_MODEL")
        .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());

    tracing::info!("loading model: {}", model_id);
    let engine = zti_embed::EmbedEngine::load(&model_id)?;
    let hw = zti_hw::probe();
    tracing::info!(device = ?hw.device, "hardware detected");

    let state = Arc::new(DaemonState::new(engine, hw));

    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!("daemon listening on {}", socket_path.display());

    listener::run(listener, state).await?;

    let _ = std::fs::remove_file(&pid_path);
    let _ = std::fs::remove_file(&socket_path);

    Ok(())
}
