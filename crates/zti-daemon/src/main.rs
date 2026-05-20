use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use fs2::FileExt;
use tokio::net::UnixListener;
use tracing_subscriber::EnvFilter;
use zti_embed::{EmbedEngine, OnnxVariant};
use zti_hw::Hardware;

mod handlers;
mod listener;
mod registry;
mod state;

use state::DaemonState;

#[derive(Parser)]
#[command(name = "zti-daemon", about = "Zebra tree indexer daemon")]
struct Cli {
    #[arg(short, long, default_value = "Xenova/bge-small-en-v1.5")]
    model: String,

    #[arg(long, value_enum, default_value_t = OnnxVariant::Auto)]
    variant: OnnxVariant,
}

fn main() -> Result<()> {
    let Cli { model, variant } = Cli::parse();

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

    // Writes to stderr. When run in a terminal you see logs directly; when
    // spawned by `zti_ipc_client::spawn::spawn_daemon`, stderr is redirected
    // to `~/.zebra_tree_indexer/zti-daemon.log` by the parent, so file
    // logging is preserved without duplicating the writer here.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "info,zti_daemon=debug,zti_embed=debug,\
                     zti_pipeline=debug,zti_dsl=debug,zti_store=debug",
            )
        }))
        .with_writer(std::io::stderr)
        .init();

    let socket_path = zti_common::paths::daemon_socket()?;
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    tracing::info!("loading model: {}", model);
    let hw = zti_hw::probe();
    tracing::info!(device = ?hw.device, "hardware detected");
    let engine = EmbedEngine::load_with_variant(&model, &hw, variant)?;

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
