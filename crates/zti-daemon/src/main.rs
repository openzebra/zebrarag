use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result};
use fs2::FileExt;
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
    let mut pid_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .read(true)
        .open(&pid_path)
        .with_context(|| format!("opening {}", pid_path.display()))?;

    if let Err(e) = pid_file.try_lock_exclusive() {
        eprintln!(
            "another daemon is running (cannot lock {}): {}",
            pid_path.display(),
            e
        );
        std::process::exit(1);
    }

    pid_file.set_len(0)?;
    write!(pid_file, "{}", std::process::id())?;
    pid_file.flush()?;

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

    let state = Arc::new(DaemonState::new(engine, hw, pid_file));

    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!("daemon listening on {}", socket_path.display());

    listener::run(listener, state).await?;

    let _ = std::fs::remove_file(&pid_path);
    let _ = std::fs::remove_file(&socket_path);

    Ok(())
}
