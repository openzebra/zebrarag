use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use fs2::FileExt;
use tokio::net::UnixListener;
use tracing_subscriber::EnvFilter;
use zti_embed::EmbedEngine;
use zti_hw::Hardware;

mod handlers;
mod listener;
mod registry;
mod state;

use state::DaemonState;

#[derive(Parser)]
#[command(name = "zti-daemon", about = "Zebra tree indexer daemon")]
struct Cli {
    #[arg(short, long)]
    model: String,
}

fn main() -> Result<()> {
    let Cli { model } = Cli::parse();

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

    let socket_path = zti_common::paths::daemon_socket()?;
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    tracing::info!("loading model: {}", model);
    let engine = EmbedEngine::load(&model)?;
    let hw = zti_hw::probe();
    tracing::info!(device = ?hw.device, "hardware detected");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_main(engine, hw, pid_file, pid_path, socket_path))
}

async fn async_main(
    engine: EmbedEngine,
    hw: Hardware,
    pid_file: std::fs::File,
    pid_path: std::path::PathBuf,
    socket_path: std::path::PathBuf,
) -> Result<()> {
    let state = Arc::new(DaemonState::new(engine, hw, pid_file));

    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!("daemon listening on {}", socket_path.display());

    listener::run(listener, state).await?;

    let _ = std::fs::remove_file(&pid_path);
    let _ = std::fs::remove_file(&socket_path);

    Ok(())
}
