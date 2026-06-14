use std::borrow::Cow;
use std::io::Write as _;
use std::sync::Arc;

use anyhow::{Context, Result};
use fs2::FileExt;
use tokio::net::UnixListener;
use tracing_subscriber::EnvFilter;
use zti_embed::{AnyEmbedEngine, EmbedEngine, LoadOverrides};
use zti_remote_embed::{RemoteEmbedEngine, RemoteModelInfo, RemoteProvider};

pub mod handlers;
pub mod listener;
pub mod registry;
pub mod state;
pub mod watch;

use state::DaemonState;

pub struct DaemonConfig<'a> {
    pub model: Cow<'a, str>,
    pub query_prefix: Option<&'a str>,
    pub passage_prefix: Option<&'a str>,
    pub model_dtype: Option<&'a str>,
    pub remote_api_key: Option<&'a str>,
    pub remote_dim_hint: Option<usize>,
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
    // Cap the `log` bridge: dependency debug/trace records are dropped
    // before they reach the tracing dispatcher (matching the main.rs policy).
    log::set_max_level(log::LevelFilter::Warn);

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
    let is_remote_model = RemoteProvider::from_model_id(config.model.as_ref()).is_some();
    let preloaded_engine = if is_remote_model {
        None
    } else {
        Some(AnyEmbedEngine::Local(EmbedEngine::load_with(
            &config.model,
            Arc::clone(&hw),
            &opts,
        )?))
    };
    let remote_api_key = config.remote_api_key.map(Arc::<str>::from);
    let remote_dim_hint = config.remote_dim_hint;

    // Build the runtime explicitly: a fixed, named worker pool (one per core)
    // keeps the reactor from over-provisioning and makes thread profiles
    // legible (`zti-rt-*` vs anonymous `tokio-runtime-worker`). The blocking
    // pool is left at its default bound — Lance drives its own IO through
    // `spawn_blocking`, so capping it risks starving storage operations.
    let worker_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .thread_name("zti-rt")
        .enable_all()
        .build()?;
    rt.block_on(async {
        let model_id: Arc<str> = Arc::from(config.model.as_ref());
        let engine = if let Some(engine) = preloaded_engine {
            engine
        } else if let Some((provider, remote_model)) =
            RemoteProvider::from_model_id(config.model.as_ref())
        {
            let api_key = remote_api_key.as_ref().map(Arc::clone).ok_or_else(|| {
                anyhow::anyhow!("remote API key is required for {}", provider.label())
            })?;
            let info = RemoteModelInfo {
                id: remote_model.to_string(),
                name: remote_model.to_string(),
                description: String::new(),
                context_length: 0,
                pricing: None,
            };
            let remote = RemoteEmbedEngine::connect(
                provider,
                Arc::clone(&api_key),
                &info,
                remote_dim_hint,
            )
            .await?;
            tracing::info!(dim = remote.dim(), model = remote.model_id(), "remote embed engine ready");
            AnyEmbedEngine::Remote(remote)
        } else {
            anyhow::bail!("unsupported remote model '{}': missing provider prefix", config.model);
        };
        let model_dtype = config.model_dtype.map(String::from);
        let state = Arc::new(DaemonState::new(
            engine,
            model_id,
            hw,
            model_dtype,
            remote_api_key,
            remote_dim_hint,
            pid_file,
        ));

        match watch::WatchManager::start(Arc::clone(&state)) {
            Ok(manager) => {
                let _ = state.watch.set(manager);
            }
            Err(e) => tracing::warn!("file watcher disabled: {e}"),
        }

        let listener = UnixListener::bind(&socket_path)?;
        tracing::info!("daemon listening on {}", socket_path.display());
        listener::run(listener, state).await?;
        let _ = std::fs::remove_file(&pid_path);
        let _ = std::fs::remove_file(&socket_path);
        Ok(())
    })
}
