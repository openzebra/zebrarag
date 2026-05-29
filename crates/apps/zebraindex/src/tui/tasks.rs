use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Mutex, mpsc};
use zti_protocol::request::{CancelIndexReq, Request};
use zti_protocol::response::Response;

use super::app::{self, DEFAULT_DIM};
use super::config;
use super::registry;

const REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/hicaru/zebra_tree_indexer/refs/heads/master/models.toml";

pub struct ClientCtx {
    pub client: Arc<Mutex<Option<zti_ipc_client::Client>>>,
    pub model: Option<Arc<str>>,
    pub query_prefix: Option<Arc<str>>,
    pub passage_prefix: Option<Arc<str>>,
    pub model_dtype: Option<Arc<str>>,
    pub search_method: Option<zti_ann::SearchMethod>,
}

impl ClientCtx {
    pub fn from_app(app: &app::App) -> Self {
        Self {
            client: app.client.clone(),
            model: app.model.clone(),
            query_prefix: app.query_prefix.clone(),
            passage_prefix: app.passage_prefix.clone(),
            model_dtype: app.model_dtype.clone(),
            search_method: app.search_method,
        }
    }

    pub fn deref_opts(&self) -> (Option<&str>, Option<&str>, Option<&str>, Option<&str>) {
        (
            self.model.as_deref(),
            self.query_prefix.as_deref(),
            self.passage_prefix.as_deref(),
            self.model_dtype.as_deref(),
        )
    }
}

pub fn build_change_method_modal(
    project_root: Option<Arc<str>>,
    canonical_path: Option<Arc<str>>,
    is_reindex: bool,
    already_indexed: Option<bool>,
    current: Option<zti_ann::SearchMethod>,
    projects: &[zti_store::ProjectRow],
    hw: &zti_hw::Hardware,
) -> app::Modal {
    let max_chunks = projects
        .iter()
        .map(|p| p.total_chunks as usize)
        .max()
        .unwrap_or(5_000);
    let recommended = zti_ann::recommend(max_chunks, DEFAULT_DIM, hw);
    let methods: Arc<[(zti_ann::SearchMethod, bool)]> =
        Arc::from(zti_ann::SearchMethod::ALL.map(|m| (m, m == recommended)));
    let selected = current
        .and_then(|c| methods.iter().position(|(m, _)| *m == c))
        .or_else(|| methods.iter().position(|(_, r)| *r))
        .unwrap_or(0);
    app::Modal::ChangeIndexMethod {
        project_root,
        canonical_path,
        is_reindex,
        already_indexed,
        methods,
        selected,
        selected_button: app::IndexMethodButton::default(),
    }
}

pub async fn resolve_startup(tx: mpsc::Sender<app::AppMessage>) {
    if let Ok(Some(cfg)) = config::load()
        && registry::is_model_downloaded(&cfg.default_model)
    {
        let _ = tx
            .send(app::AppMessage::ConfigResolved {
                model: Some(cfg.default_model),
                search_method: cfg.default_search_method,
                model_dtype: cfg.default_dtype,
            })
            .await;
        return;
    }

    if let Ok(projects) = zti_store::list_projects().await
        && let Some(p) = projects
            .into_iter()
            .filter(|p| !p.model_id.is_empty())
            .max_by_key(|p| p.last_indexed_ns)
        && registry::is_model_downloaded(&p.model_id)
    {
        let _ = config::save(&p.model_id, None, None);
        let _ = tx
            .send(app::AppMessage::ConfigResolved {
                model: Some(p.model_id),
                search_method: None,
                model_dtype: None,
            })
            .await;
        return;
    }

    let _ = tx
        .send(app::AppMessage::ConfigResolved {
            model: None,
            search_method: None,
            model_dtype: None,
        })
        .await;
}

pub async fn fetch_registry(tx: mpsc::Sender<app::AppMessage>) {
    if let Ok(Some(reg)) = registry::load() {
        let _ = tx.send(app::AppMessage::RegistryLoaded(reg.entries)).await;
        return;
    }

    let result: anyhow::Result<Vec<registry::ModelEntry>> = async {
        let resp = reqwest::get(REGISTRY_URL).await?;
        let body = resp.text().await?;
        let path = registry::registry_path()?;
        tokio::fs::write(&path, body.as_bytes()).await?;
        let reg = registry::parse(&body)?;
        Ok(reg.entries)
    }
    .await;

    match result {
        Ok(entries) => {
            let _ = tx.send(app::AppMessage::RegistryLoaded(entries)).await;
        }
        Err(e) => {
            let _ = tx.send(app::AppMessage::RegistryError(e.to_string())).await;
        }
    }
}

pub async fn download_model(model_id: Arc<str>, tx: mpsc::Sender<app::AppMessage>) {
    let id = Arc::clone(&model_id);
    let result =
        tokio::task::spawn_blocking(move || zti_embed::model_registry::resolve_model_files(&id))
            .await;

    match result {
        Ok(Ok(_)) => {
            let _ = tx.send(app::AppMessage::ModelDownloaded(model_id)).await;
        }
        Ok(Err(e)) => {
            let _ = tx.send(app::AppMessage::ModelDownloadError(e.to_string())).await;
        }
        Err(e) => {
            let _ = tx.send(app::AppMessage::ModelDownloadError(e.to_string())).await;
        }
    }
}

pub fn spawn_refresh_projects(tx: &mpsc::Sender<app::AppMessage>) {
    let tx_c = tx.clone();
    tokio::spawn(async move {
        if let Ok(projects) = zti_store::list_projects().await {
            let _ = tx_c.send(app::AppMessage::ProjectsLoaded(projects)).await;
        }
    });
}

pub fn spawn_daemon_monitor(app: &mut app::App, tx: &mpsc::Sender<app::AppMessage>) {
    if let Some(handle) = app.monitor_handle.take() {
        handle.abort();
    }
    let ctx = ClientCtx::from_app(app);
    let tx_m = tx.clone();
    let should_run = app.should_run.clone();
    let handle = tokio::spawn(async move {
        daemon_monitor(tx_m, ctx, should_run).await;
    });
    app.monitor_handle = Some(handle);
}

async fn ensure_client(
    client: &Arc<Mutex<Option<zti_ipc_client::Client>>>,
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
    model_dtype: Option<&str>,
) -> anyhow::Result<()> {
    let mut guard = client.lock().await;
    if guard.is_none() {
        let mut c = zti_ipc_client::Client::connect(
            Duration::from_secs(10),
            model,
            query_prefix,
            passage_prefix,
            model_dtype,
        )
        .await?;
        c.handshake().await?;
        *guard = Some(c);
    }
    Ok(())
}

fn read_daemon_log_tail(msg: &mut String) {
    if let Ok(log_path) = zti_common::paths::daemon_log()
        && let Ok(log) = std::fs::read_to_string(&log_path)
    {
        let mut lines = Vec::with_capacity(5);
        lines.extend(log.lines().rev().take(5));
        lines.reverse();
        let tail: String = lines.join("\n");
        if !tail.is_empty() {
            msg.push_str("\n\ndaemon.log:\n");
            msg.push_str(&tail);
        }
    }
}

async fn try_connect(ctx: &ClientCtx, tx: &mpsc::Sender<app::AppMessage>) {
    let (m, qp, pp, md) = ctx.deref_opts();
    if let Err(e) = ensure_client(&ctx.client, m, qp, pp, md).await {
        let mut msg = e.to_string();
        read_daemon_log_tail(&mut msg);
        let _ = tx
            .send(app::AppMessage::DaemonStatusUpdate(app::DaemonStatus::Error(msg)))
            .await;
    }
}

async fn daemon_monitor(
    tx: mpsc::Sender<app::AppMessage>,
    ctx: ClientCtx,
    should_run: Arc<AtomicBool>,
) {
    let mut env_fetched = false;
    loop {
        let socket_path = match zti_common::paths::daemon_socket() {
            Ok(p) => p,
            Err(_) => {
                let _ = tx
                    .send(app::AppMessage::DaemonStatusUpdate(app::DaemonStatus::Error(
                        "cannot resolve socket path".into(),
                    )))
                    .await;
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        if !socket_path.exists() {
            if !should_run.load(Ordering::Relaxed) {
                let _ = tx
                    .send(app::AppMessage::DaemonStatusUpdate(app::DaemonStatus::Stopped))
                    .await;
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
            let _ = tx
                .send(app::AppMessage::DaemonStatusUpdate(app::DaemonStatus::Starting))
                .await;
            try_connect(&ctx, &tx).await;
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        }

        let status = fetch_daemon_status(&ctx).await;
        if let Some(s) = status {
            let _ = tx.send(app::AppMessage::DaemonStatusUpdate(s)).await;
            if !env_fetched
                && let Some(env_info) = fetch_daemon_env(&ctx).await
            {
                let _ = tx
                    .send(app::AppMessage::DaemonEnvLoaded {
                        cpus: env_info.cpus,
                        mem_total_mb: env_info.mem_total_mb,
                    })
                    .await;
                env_fetched = true;
            }
        } else {
            let _ = tx
                .send(app::AppMessage::DaemonStatusUpdate(app::DaemonStatus::Starting))
                .await;
            try_connect(&ctx, &tx).await;
        }

        if let Ok(projects) = zti_store::list_projects().await {
            let _ = tx.send(app::AppMessage::ProjectsLoaded(projects)).await;
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn fetch_daemon_status(
    ctx: &ClientCtx,
) -> Option<app::DaemonStatus> {
    let mut guard = ctx.client.lock().await;
    match guard.as_mut() {
        Some(c) => match c.request(Request::DaemonStatus).await {
            Ok(Response::DaemonStatus(info)) => Some(app::DaemonStatus::Running {
                device: info.device,
                uptime_secs: info.uptime_secs,
                cpus: info.cpus,
                mem_total_mb: info.mem_total_mb,
            }),
            Ok(_) => None,
            Err(e) => {
                *guard = None;
                Some(app::DaemonStatus::Error(e.to_string()))
            }
        },
        None => None,
    }
}

async fn fetch_daemon_env(
    ctx: &ClientCtx,
) -> Option<zti_protocol::response::DaemonEnvInfo> {
    let mut guard = ctx.client.lock().await;
    match guard.as_mut() {
        Some(c) => match c.request(Request::DaemonEnv).await {
            Ok(Response::DaemonEnv(info)) => Some(info),
            _ => None,
        },
        None => None,
    }
}

pub async fn do_search(
    query: String,
    mode: zti_protocol::request::SearchMode,
    root: Option<String>,
    ctx: ClientCtx,
    tx: mpsc::Sender<app::AppMessage>,
) {
    let result = async {
        let (m, qp, pp, md) = ctx.deref_opts();
        ensure_client(&ctx.client, m, qp, pp, md).await?;

        let project_root = match root {
            Some(r) => r,
            None => {
                let projects = zti_store::list_projects().await?;
                match projects.into_iter().next() {
                    Some(p) => p.root_path,
                    None => anyhow::bail!("No indexed projects"),
                }
            }
        };

        let mut guard = ctx.client.lock().await;
        let c = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("client not initialized"))?;

        let resp = c
            .request(Request::Search(zti_protocol::request::SearchReq {
                project_root,
                query,
                limit: 10,
                offset: None,
                languages: None,
                path_glob: None,
                refresh_index: false,
                exhaustive: false,
                mode,
            }))
            .await?;

        match resp {
            Response::Search(Ok(results)) => Ok(results),
            Response::Search(Err(e)) => Err(anyhow::anyhow!(e.message)),
            other => Err(anyhow::anyhow!("unexpected: {:?}", other)),
        }
    }
    .await;

    match result {
        Ok(results) => {
            let _ = tx.send(app::AppMessage::SearchDone(results)).await;
        }
        Err(e) => {
            let _ = tx.send(app::AppMessage::SearchError(e.to_string())).await;
        }
    }
}

pub async fn do_remove_project(
    project_root: String,
    project_id: [u8; 32],
    ctx: ClientCtx,
    tx: mpsc::Sender<app::AppMessage>,
) {
    let daemon_err = async {
        let (m, qp, pp, md) = ctx.deref_opts();
        ensure_client(&ctx.client, m, qp, pp, md).await?;

        let mut guard = ctx.client.lock().await;
        let c = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("client not initialized"))?;

        let resp = c
            .request(Request::RemoveProject(
                zti_protocol::request::RemoveProjectReq { project_root },
            ))
            .await?;

        match resp {
            Response::RemoveProject(Ok(())) => Ok(()),
            Response::RemoveProject(Err(e)) => Err(anyhow::anyhow!(e.message)),
            other => Err(anyhow::anyhow!("unexpected: {:?}", other)),
        }
    }
    .await
    .err();

    if let Ok(dir) = zti_common::paths::project_dir_path(&project_id)
        && dir.exists()
        && let Err(e) = std::fs::remove_dir_all(&dir)
    {
        let msg = daemon_err
            .map(|de| format!("{de}; disk: {e}"))
            .unwrap_or_else(|| format!("failed to delete project data: {e}"));
        let _ = tx.send(app::AppMessage::ProjectRemoveError(msg)).await;
        return;
    }

    let _ = tx.send(app::AppMessage::ProjectRemoved).await;
}

#[derive(PartialEq, Eq, Debug)]
pub enum IndexMode {
    Initial,
    Reindex,
    ForceReindex,
}

pub async fn do_index(
    project_root: String,
    mode: IndexMode,
    ctx: ClientCtx,
    tx: mpsc::Sender<app::AppMessage>,
) {
    let result = async {
        let (m, qp, pp, md) = ctx.deref_opts();
        ensure_client(&ctx.client, m, qp, pp, md).await?;

        let mut guard = ctx.client.lock().await;
        let c = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("client not initialized"))?;

        let tx_p = tx.clone();
        let resp = c
            .request_streaming(
                Request::Index(zti_protocol::request::IndexReq {
                    project_root,
                    refresh: matches!(mode, IndexMode::ForceReindex),
                    search_method: ctx.search_method.map(|m| m.as_str().to_string()),
                }),
                |frame| {
                    if let Response::IndexProgress(p) = frame {
                        if tx_p.try_send(app::AppMessage::IndexProgress {
                            phase: p.phase,
                            current: p.current,
                            total: p.total,
                            message: p.message,
                            is_reindex: matches!(mode, IndexMode::Reindex | IndexMode::ForceReindex),
                        }).is_err() {
                            tracing::warn!("dropped index progress frame");
                        }
                    }
                },
            )
            .await?;

        match resp {
            Response::Index(Ok(_)) => Ok(()),
            Response::Index(Err(e)) => Err(anyhow::anyhow!(e.message)),
            other => Err(anyhow::anyhow!("unexpected: {:?}", other)),
        }
    }
    .await;

    match result {
        Ok(()) => {
            let _ = tx.send(app::AppMessage::IndexComplete).await;
        }
        Err(e) => {
            let msg = e.to_string();
            if msg == "indexing cancelled" {
                let _ = tx.send(app::AppMessage::IndexCancelled).await;
            } else {
                let _ = tx.send(app::AppMessage::IndexError(msg)).await;
            }
        }
    }
}

pub async fn cancel_index(project_root: String, ctx: ClientCtx) {
    let (m, qp, pp, md) = ctx.deref_opts();
    let mut client = match zti_ipc_client::Client::connect(
        Duration::from_secs(10),
        m,
        qp,
        pp,
        md,
    )
    .await
    {
        Ok(c) => c,
        Err(_) => return,
    };
    if client.handshake().await.is_err() {
        return;
    }
    let _ = client
        .request(Request::CancelIndex(CancelIndexReq { project_root }))
        .await;
}
