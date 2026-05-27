use std::borrow::Cow;
use std::io::Write as _;
use std::sync::Arc;

use anyhow::{Context, Result};
use fs2::FileExt;
use tokio::net::UnixListener;
use tracing_subscriber::EnvFilter;
use zti_embed::{EmbedEngine, LoadOverrides};

pub mod handlers;
pub mod listener;
pub mod registry;
pub mod state;

use state::DaemonState;

pub struct DaemonConfig<'a> {
    pub model: Cow<'a, str>,
    pub query_prefix: Option<&'a str>,
    pub passage_prefix: Option<&'a str>,
    pub model_dtype: Option<&'a str>,
}

pub fn run_daemon(config: &DaemonConfig<'_>) -> Result<()> {
    let pid_path = zti_common::paths::daemon_pid()?;
    let mut pid_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .read(true)
        .open(&pid_path)
        .with_context(|| format!("opening {}", pid_path.display()))?;

    if let Err(e) = pid_file.try_lock_exclusive() {
        anyhow::bail!(
            "another daemon is running (cannot lock {}): {}",
            pid_path.display(),
            e
        );
    }

    pid_file.set_len(0)?;
    write!(pid_file, "{}", std::process::id())?;
    pid_file.flush()?;

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "info,zti_daemon=debug,zti_embed=debug,\
                 zti_pipeline=debug,zti_dsl=debug,zti_store=debug,\
                 zti_rerank=trace",
            )
        }))
        .with_writer(std::io::stderr)
        .init();

    let socket_path = zti_common::paths::daemon_socket()?;
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    tracing::info!("loading model: {}", config.model);
    let hw = Arc::new(zti_hw::probe());
    tracing::info!(device = ?hw.device, "hardware detected");

    let opts = LoadOverrides {
        query_prefix: config.query_prefix,
        passage_prefix: config.passage_prefix,
        model_dtype: config.model_dtype.and_then(zti_embed::parse_model_dtype),
    };
    let engine = EmbedEngine::load_with(&config.model, Arc::clone(&hw), &opts)?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let model_id: Arc<str> = Arc::from(config.model.as_ref());
        let state = Arc::new(DaemonState::new(engine, model_id, hw, pid_file));
        let listener = UnixListener::bind(&socket_path)?;
        tracing::info!("daemon listening on {}", socket_path.display());
        listener::run(listener, state).await?;
        let _ = std::fs::remove_file(&pid_path);
        let _ = std::fs::remove_file(&socket_path);
        Ok(())
    })
}
