mod app;
mod config;
mod event;
mod registry;
mod setup;
mod ui;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::{Mutex, mpsc};
use zti_protocol::request::Request;
use zti_protocol::response::Response;

use app::{App, AppMessage};

const DEFAULT_DIM: usize = 768;

const REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/hicaru/zebra_tree_indexer/refs/heads/master/models.toml";

pub fn run_tui(
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
    model_dtype: Option<&str>,
) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        terminal::enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut app = App {
            model: model.map(Arc::from),
            query_prefix: query_prefix.map(Arc::from),
            passage_prefix: passage_prefix.map(Arc::from),
            model_dtype: model_dtype.map(Arc::from),
            ..App::default()
        };

        let (tx, mut rx) = mpsc::channel::<AppMessage>(32);

        if model.is_some() {
            app.screen = app::Screen::Main;
            spawn_daemon_monitor(&mut app, &tx);
        } else {
            let tx_r = tx.clone();
            tokio::spawn(async move { resolve_startup(tx_r).await });
        }

        let mut tick: u16 = 0;

        loop {
            terminal.draw(|f| match &app.screen {
                app::Screen::Setup(phase) => setup::draw(f, phase, tick),
                app::Screen::Main => ui::draw(f, &app, tick),
            })?;

            while let Ok(msg) = rx.try_recv() {
                dispatch(&mut app, msg, &tx).await;
            }

            if crossterm::event::poll(Duration::from_millis(50))?
                && let crossterm::event::Event::Key(key) = crossterm::event::read()?
            {
                let action = event::map_key(&key, &app);
                handle_action(&mut app, action, &tx).await;
            }

            tick = tick.wrapping_add(1);

            if app.should_quit {
                break;
            }
        }

        terminal::disable_raw_mode()?;
        execute!(std::io::stdout(), LeaveAlternateScreen)?;
        Ok(())
    })
}

async fn resolve_startup(tx: mpsc::Sender<AppMessage>) {
    if let Ok(Some(cfg)) = config::load()
        && registry::is_model_downloaded(&cfg.default_model)
    {
        let _ = tx
            .send(AppMessage::ConfigResolved {
                model: Some(cfg.default_model),
                search_method: cfg.default_search_method,
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
        let _ = config::save(&p.model_id, None);
        let _ = tx
            .send(AppMessage::ConfigResolved {
                model: Some(p.model_id),
                search_method: None,
            })
            .await;
        return;
    }

    let _ = tx
        .send(AppMessage::ConfigResolved {
            model: None,
            search_method: None,
        })
        .await;
}

async fn fetch_registry(tx: mpsc::Sender<AppMessage>) {
    if let Ok(Some(reg)) = registry::load() {
        let _ = tx.send(AppMessage::RegistryLoaded(reg.entries)).await;
        return;
    }

    let result: Result<Vec<registry::ModelEntry>> = async {
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
            let _ = tx.send(AppMessage::RegistryLoaded(entries)).await;
        }
        Err(e) => {
            let _ = tx.send(AppMessage::RegistryError(e.to_string())).await;
        }
    }
}

async fn download_model(model_id: Arc<str>, tx: mpsc::Sender<AppMessage>) {
    let id = Arc::clone(&model_id);
    let result =
        tokio::task::spawn_blocking(move || zti_embed::model_registry::resolve_model_files(&id))
            .await;

    match result {
        Ok(Ok(_)) => {
            let _ = tx.send(AppMessage::ModelDownloaded(model_id)).await;
        }
        Ok(Err(e)) => {
            let _ = tx.send(AppMessage::ModelDownloadError(e.to_string())).await;
        }
        Err(e) => {
            let _ = tx.send(AppMessage::ModelDownloadError(e.to_string())).await;
        }
    }
}

async fn dispatch(app: &mut App, msg: AppMessage, tx: &mpsc::Sender<AppMessage>) {
    match msg {
        AppMessage::ConfigResolved { model: None, .. } => {
            app.screen = app::Screen::Setup(app::SetupPhase::FetchingRegistry);
            let tx_c = tx.clone();
            tokio::spawn(async move { fetch_registry(tx_c).await });
        }
        AppMessage::ConfigResolved {
            model: Some(m),
            search_method,
        } => {
            app.model = Some(Arc::from(m.as_str()));
            app.search_method = search_method
                .as_deref()
                .and_then(zti_ann::SearchMethod::parse);
            app.should_run.store(true, Ordering::Relaxed);
            app.screen = app::Screen::Main;
            spawn_daemon_monitor(app, tx);
        }
        AppMessage::RegistryLoaded(mut entries) => {
            registry::sort_by_hardware(&mut entries);
            let shared: Arc<[registry::ModelEntry]> = entries.into();
            app.setup_registry = Some(Arc::clone(&shared));
            app.screen = app::Screen::Setup(app::SetupPhase::ModelSelection {
                entries: shared,
                selected: 0,
            });
        }
        AppMessage::RegistryError(msg) => {
            app.screen = app::Screen::Setup(app::SetupPhase::Error {
                message: msg,
                can_retry: true,
            });
        }
        AppMessage::ModelDownloaded(model_id) => {
            let _ = tx.send(AppMessage::SetupComplete { model: model_id }).await;
        }
        AppMessage::ModelDownloadError(msg) => {
            app.screen = app::Screen::Setup(app::SetupPhase::Error {
                message: msg,
                can_retry: false,
            });
        }
        AppMessage::SetupComplete { model } => {
            app.model = Some(Arc::clone(&model));
            app.should_run.store(true, Ordering::Relaxed);
            app.screen = app::Screen::Main;
            spawn_daemon_monitor(app, tx);
        }
        AppMessage::ProjectRemoved => {
            app.modal = None;
            if app.selected_project < app.projects.len() {
                app.projects.remove(app.selected_project);
                if app.selected_project >= app.projects.len() && !app.projects.is_empty() {
                    app.selected_project = app.projects.len() - 1;
                }
            }
            spawn_refresh_projects(tx);
        }
        AppMessage::ProjectRemoveError(e) => {
            app.modal = Some(app::Modal::Error { message: e });
        }
        AppMessage::IndexComplete => {
            app.modal = None;
            spawn_refresh_projects(tx);
        }
        AppMessage::IndexProgress {
            current,
            total,
            message,
            is_reindex,
        } => {
            app.modal = Some(app::Modal::Indexing {
                current,
                total,
                message,
                is_reindex,
            });
        }
        AppMessage::IndexError(e) => {
            app.modal = Some(app::Modal::Error { message: e });
        }
        other => app.apply_message(other),
    }
}

fn spawn_refresh_projects(tx: &mpsc::Sender<AppMessage>) {
    let tx_c = tx.clone();
    tokio::spawn(async move {
        if let Ok(projects) = zti_store::list_projects().await {
            let _ = tx_c.send(AppMessage::ProjectsLoaded(projects)).await;
        }
    });
}

fn spawn_daemon_monitor(app: &mut App, tx: &mpsc::Sender<AppMessage>) {
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

async fn handle_action(app: &mut App, action: event::Action, tx: &mpsc::Sender<AppMessage>) {
    match action {
        event::Action::Quit => app.should_quit = true,

        event::Action::SetupNext => {
            match &mut app.screen {
                app::Screen::Setup(app::SetupPhase::ModelSelection { entries, selected })
                    if *selected + 1 < entries.len() =>
                {
                    *selected += 1;
                }
                app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                    methods,
                    selected,
                    ..
                }) if *selected + 1 < methods.len() => {
                    *selected += 1;
                }
                _ => {}
            }
            if let Some(app::Modal::ChangeIndexMethod {
                methods, selected, ..
            }) = &mut app.modal
                && *selected + 1 < methods.len()
            {
                *selected += 1;
            }
        }
        event::Action::SetupPrev => {
            if let app::Screen::Setup(
                app::SetupPhase::ModelSelection { selected, .. }
                | app::SetupPhase::IndexMethodSelection { selected, .. },
            ) = &mut app.screen
                && *selected > 0
            {
                *selected -= 1;
            }
            if let Some(app::Modal::ChangeIndexMethod { selected, .. }) = &mut app.modal
                && *selected > 0
            {
                *selected -= 1;
            }
        }
        event::Action::SetupAutoRecommend => {
            if let app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                methods,
                selected,
                ..
            }) = &mut app.screen
                && let Some(pos) = methods.iter().position(|(_, r)| *r)
            {
                *selected = pos;
            }
            if let Some(app::Modal::ChangeIndexMethod {
                methods, selected, ..
            }) = &mut app.modal
                && let Some(pos) = methods.iter().position(|(_, r)| *r)
            {
                *selected = pos;
            }
        }
        event::Action::SetupConfirm => match &app.screen {
            app::Screen::Setup(app::SetupPhase::ModelSelection { entries, selected }) => {
                let entry = &entries[*selected];
                let model_id: Arc<str> = Arc::from(entry.model_id.as_str());
                if entry.is_downloaded() {
                    let hw = zti_hw::probe();
                    let max_chunks = app
                        .projects
                        .iter()
                        .map(|p| p.total_chunks as usize)
                        .max()
                        .unwrap_or(5_000);
                    let recommended = zti_ann::recommend(max_chunks, DEFAULT_DIM, &hw);
                    let methods: Arc<[(zti_ann::SearchMethod, bool)]> =
                        Arc::from(zti_ann::SearchMethod::ALL.map(|m| (m, m == recommended)));
                    let rec_idx = methods.iter().position(|(_, r)| *r).unwrap_or(0);
                    app.screen = app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                        model_id,
                        methods,
                        selected: rec_idx,
                    });
                } else {
                    let id = Arc::clone(&model_id);
                    app.screen = app::Screen::Setup(app::SetupPhase::DownloadingModel { model_id });
                    let tx_c = tx.clone();
                    tokio::spawn(async move { download_model(id, tx_c).await });
                }
            }
            app::Screen::Setup(app::SetupPhase::IndexMethodSelection {
                model_id,
                methods,
                selected,
            }) => {
                let (method, _) = methods[*selected];
                app.search_method = Some(method);
                let save_model = Arc::clone(model_id);
                let launch_model = Arc::clone(model_id);
                let complete_model = Arc::clone(model_id);

                if let Err(e) = config::save(&save_model, Some(method.as_str())) {
                    app.screen = app::Screen::Setup(app::SetupPhase::Error {
                        message: format!("Failed to save config: {e}"),
                        can_retry: false,
                    });
                    return;
                }

                app.screen = app::Screen::Setup(app::SetupPhase::Launching {
                    model_id: launch_model,
                });

                let _ = tx
                    .send(AppMessage::SetupComplete {
                        model: complete_model,
                    })
                    .await;
            }
            _ => {}
        },
        event::Action::SetupBack => {
            if let app::Screen::Setup(app::SetupPhase::ModelSelection { .. }) = &app.screen {
                if app.model.is_some() {
                    app.should_run.store(true, Ordering::Relaxed);
                    app.screen = app::Screen::Main;
                } else {
                    app.should_quit = true;
                }
            }
        }
        event::Action::SetupRetry => {
            app.screen = app::Screen::Setup(app::SetupPhase::FetchingRegistry);
            let tx_c = tx.clone();
            tokio::spawn(async move { fetch_registry(tx_c).await });
        }

        event::Action::SwitchPanel => {
            app.active_panel = match app.active_panel {
                app::ActivePanel::Projects => app::ActivePanel::Search,
                app::ActivePanel::Search => app::ActivePanel::Projects,
            };
        }
        event::Action::FocusSearch => {
            app.active_panel = app::ActivePanel::Search;
            app.active_input = 0;
        }
        event::Action::SelectPrevProject => {
            if app.selected_project > 0 {
                app.selected_project -= 1;
            }
        }
        event::Action::SelectNextProject => {
            if app.selected_project < app.projects.len() {
                app.selected_project += 1;
            }
        }
        event::Action::ScrollUp => {
            if app.results_scroll > 0 {
                app.results_scroll -= 1;
            }
        }
        event::Action::ScrollDown => {
            app.results_scroll = app.results_scroll.saturating_add(1);
        }
        event::Action::Input(c) => {
            if let Some(app::Modal::AddProject {
                ref mut path_input,
                ref mut error,
                ..
            }) = app.modal
            {
                path_input.push(c);
                *error = None;
            } else {
                app.active_search_mut().text.push(c);
            }
        }
        event::Action::Backspace => {
            if let Some(app::Modal::AddProject {
                ref mut path_input, ..
            }) = app.modal
            {
                path_input.pop();
            } else {
                app.active_search_mut().text.pop();
            }
        }
        event::Action::SubmitSearch => {
            if app.active_search().text.is_empty() || app.searching {
                return;
            }
            let mode = app.active_search().mode;
            let query = std::mem::take(&mut app.search_inputs[app.active_input].text);
            app.searching = true;
            app.search_error = None;
            let root = app.selected_project_root().map(|s| s.to_string());
            let ctx = ClientCtx::from_app(app);
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                do_search(query, mode, root, ctx, tx_clone).await;
            });
        }
        event::Action::ToggleSearchInput => {
            app.active_input ^= 1;
        }
        event::Action::StopDaemon => {
            app.should_run.store(false, Ordering::Relaxed);
            let client = app.client.clone();
            tokio::spawn(async move {
                let mut guard = client.lock().await;
                if let Some(mut c) = guard.take() {
                    let _ = c.request(Request::Stop).await;
                }
            });
        }
        event::Action::RestartDaemon => {
            app.should_run.store(true, Ordering::Relaxed);
            {
                let mut guard = app.client.lock().await;
                *guard = None;
            }
            app.daemon_status = app::DaemonStatus::Starting;
        }
        event::Action::ChangeModel => {
            app.should_run.store(false, Ordering::Relaxed);
            let client = app.client.clone();
            tokio::spawn(async move {
                let mut guard = client.lock().await;
                if let Some(mut c) = guard.take() {
                    let _ = c.request(Request::Stop).await;
                }
            });
            app.screen = app::Screen::Setup(app::SetupPhase::FetchingRegistry);
            let tx_c = tx.clone();
            tokio::spawn(async move { fetch_registry(tx_c).await });
        }

        event::Action::OpenProjectDetail => {
            if app.selected_project == app.projects.len() {
                app.modal = Some(app::Modal::AddProject {
                    path_input: String::with_capacity(256),
                    error: None,
                });
            } else if !app.projects.is_empty() {
                app.modal = Some(app::Modal::ProjectDetail {
                    selected_button: app::DetailButton::default(),
                });
            }
        }
        event::Action::DetailButtonNext => match &mut app.modal {
            Some(app::Modal::ProjectDetail { selected_button }) => {
                *selected_button = match selected_button {
                    app::DetailButton::Remove => app::DetailButton::Reindex,
                    app::DetailButton::Reindex => app::DetailButton::Back,
                    app::DetailButton::Back => app::DetailButton::Remove,
                };
            }
            Some(app::Modal::ChangeIndexMethod {
                selected_button,
                canonical_path,
                ..
            }) if canonical_path.is_some() => {
                *selected_button = match selected_button {
                    app::IndexMethodButton::Confirm => app::IndexMethodButton::Cancel,
                    app::IndexMethodButton::Cancel => app::IndexMethodButton::Confirm,
                };
            }
            _ => {}
        },
        event::Action::DetailButtonPrev => match &mut app.modal {
            Some(app::Modal::ProjectDetail { selected_button }) => {
                *selected_button = match selected_button {
                    app::DetailButton::Remove => app::DetailButton::Back,
                    app::DetailButton::Reindex => app::DetailButton::Remove,
                    app::DetailButton::Back => app::DetailButton::Reindex,
                };
            }
            Some(app::Modal::ChangeIndexMethod {
                selected_button,
                canonical_path,
                ..
            }) if canonical_path.is_some() => {
                *selected_button = match selected_button {
                    app::IndexMethodButton::Confirm => app::IndexMethodButton::Cancel,
                    app::IndexMethodButton::Cancel => app::IndexMethodButton::Confirm,
                };
            }
            _ => {}
        },
        event::Action::DetailConfirm => match app.modal.take() {
            Some(app::Modal::ProjectDetail { selected_button }) => match selected_button {
                app::DetailButton::Remove => {
                    app.modal = Some(app::Modal::ConfirmRemove);
                }
                app::DetailButton::Reindex => {
                    if let Some(root) = app.selected_project_root() {
                        let root = root.to_string();
                        let ctx = ClientCtx::from_app(app);
                        let tx_c = tx.clone();
                        tokio::spawn(async move {
                            do_index(root, true, ctx, tx_c).await;
                        });
                    }
                }
                app::DetailButton::Back => {}
            },
            Some(app::Modal::ChangeIndexMethod {
                project_root,
                canonical_path,
                is_reindex,
                methods,
                selected,
                selected_button,
                ..
            }) => {
                if let Some(cp) = canonical_path {
                    match selected_button {
                        app::IndexMethodButton::Cancel => {
                            app.modal = Some(app::Modal::AddProject {
                                path_input: cp,
                                error: None,
                            });
                            return;
                        }
                        app::IndexMethodButton::Confirm => {}
                    }
                }
                let (method, _) = methods[selected];
                app.search_method = Some(method);
                if let Err(e) =
                    config::save(app.model.as_deref().unwrap_or(""), Some(method.as_str()))
                {
                    app.modal = Some(app::Modal::Error {
                        message: format!("Failed to save config: {e}"),
                    });
                    return;
                }
                if let Some(root) = project_root {
                    app.modal = Some(app::Modal::Indexing {
                        current: 0,
                        total: 0,
                        message: String::with_capacity(64),
                        is_reindex,
                    });
                    let ctx = ClientCtx::from_app(app);
                    let tx_c = tx.clone();
                    let refresh = is_reindex;
                    tokio::spawn(async move {
                        do_index(root, refresh, ctx, tx_c).await;
                    });
                }
            }
            other => {
                app.modal = other;
            }
        },
        event::Action::DetailBack => {
            app.modal = None;
        }
        event::Action::ConfirmRemoveYes => {
            app.modal = None;
            if let Some(p) = app.projects.get(app.selected_project)
                && let Ok(pid) = <[u8; 32]>::try_from(p.project_id.as_slice())
            {
                let root = p.root_path.clone();
                let ctx = ClientCtx::from_app(app);
                let tx_c = tx.clone();
                tokio::spawn(async move {
                    do_remove_project(root, pid, ctx, tx_c).await;
                });
            }
        }
        event::Action::ConfirmRemoveNo => {
            if app.projects.get(app.selected_project).is_some() {
                app.modal = Some(app::Modal::ProjectDetail {
                    selected_button: app::DetailButton::Remove,
                });
            } else {
                app.modal = None;
            }
        }
        event::Action::SubmitPath => {
            if let Some(app::Modal::AddProject { ref path_input, .. }) = app.modal {
                let trimmed = path_input.trim();
                if trimmed.is_empty() {
                    if let Some(app::Modal::AddProject { ref mut error, .. }) = app.modal {
                        *error = Some(String::from("Path cannot be empty"));
                    }
                } else {
                    let path = std::path::Path::new(trimmed);
                    if !path.is_dir() {
                        if let Some(app::Modal::AddProject { ref mut error, .. }) = app.modal {
                            *error = Some(String::from("Directory does not exist"));
                        }
                    } else if let Ok(canonical) = path.canonicalize() {
                        let canonical_str = canonical.to_string_lossy().into_owned();
                        let already_indexed =
                            app.projects.iter().any(|p| p.root_path == canonical_str);
                        app.modal = Some(build_change_method_modal(
                            Some(canonical_str.clone()),
                            Some(canonical_str),
                            false,
                            Some(already_indexed),
                            app.search_method,
                            &app.projects,
                        ));
                    } else if let Some(app::Modal::AddProject { ref mut error, .. }) = app.modal {
                        *error = Some(String::from("Cannot resolve path"));
                    }
                }
            }
        }
        event::Action::None => {}
    }
}

struct ClientCtx {
    client: Arc<Mutex<Option<zti_ipc_client::Client>>>,
    model: Option<Arc<str>>,
    query_prefix: Option<Arc<str>>,
    passage_prefix: Option<Arc<str>>,
    model_dtype: Option<Arc<str>>,
    search_method: Option<zti_ann::SearchMethod>,
}

impl ClientCtx {
    fn from_app(app: &App) -> Self {
        Self {
            client: app.client.clone(),
            model: app.model.clone(),
            query_prefix: app.query_prefix.clone(),
            passage_prefix: app.passage_prefix.clone(),
            model_dtype: app.model_dtype.clone(),
            search_method: app.search_method,
        }
    }

    fn deref_opts(&self) -> (Option<&str>, Option<&str>, Option<&str>, Option<&str>) {
        (
            self.model.as_deref(),
            self.query_prefix.as_deref(),
            self.passage_prefix.as_deref(),
            self.model_dtype.as_deref(),
        )
    }
}

fn build_change_method_modal(
    project_root: Option<String>,
    canonical_path: Option<String>,
    is_reindex: bool,
    already_indexed: Option<bool>,
    current: Option<zti_ann::SearchMethod>,
    projects: &[zti_store::ProjectRow],
) -> app::Modal {
    let hw = zti_hw::probe();
    let max_chunks = projects
        .iter()
        .map(|p| p.total_chunks as usize)
        .max()
        .unwrap_or(5_000);
    let recommended = zti_ann::recommend(max_chunks, DEFAULT_DIM, &hw);
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
        let mut lines: Vec<&str> = log.lines().rev().take(5).collect();
        lines.reverse();
        let tail: String = lines.join("\n");
        if !tail.is_empty() {
            msg.push_str("\n\ndaemon.log:\n");
            msg.push_str(&tail);
        }
    }
}

async fn try_connect(ctx: &ClientCtx, tx: &mpsc::Sender<AppMessage>) {
    let (m, qp, pp, md) = ctx.deref_opts();
    if let Err(e) = ensure_client(&ctx.client, m, qp, pp, md).await {
        let mut msg = e.to_string();
        read_daemon_log_tail(&mut msg);
        let _ = tx
            .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Error(
                msg,
            )))
            .await;
    }
}

async fn daemon_monitor(tx: mpsc::Sender<AppMessage>, ctx: ClientCtx, should_run: Arc<AtomicBool>) {
    loop {
        let socket_path = match zti_common::paths::daemon_socket() {
            Ok(p) => p,
            Err(_) => {
                let _ = tx
                    .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Error(
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
                    .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Stopped))
                    .await;
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
            let _ = tx
                .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Starting))
                .await;
            try_connect(&ctx, &tx).await;
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        }

        let status = {
            let mut guard = ctx.client.lock().await;
            match guard.as_mut() {
                Some(c) => match c.request(Request::DaemonStatus).await {
                    Ok(Response::DaemonStatus(info)) => Some(app::DaemonStatus::Running {
                        model_id: info.model_id,
                        device: info.device,
                        uptime_secs: info.uptime_secs,
                        loaded_models: info.loaded_models,
                        loading_model: info.loading_model,
                    }),
                    Ok(_) => None,
                    Err(e) => {
                        *guard = None;
                        Some(app::DaemonStatus::Error(e.to_string()))
                    }
                },
                None => None,
            }
        };

        if let Some(s) = status {
            let _ = tx.send(AppMessage::DaemonStatusUpdate(s)).await;
        } else {
            let _ = tx
                .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Starting))
                .await;
            try_connect(&ctx, &tx).await;
        }

        if let Ok(projects) = zti_store::list_projects().await {
            let _ = tx.send(AppMessage::ProjectsLoaded(projects)).await;
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn do_search(
    query: String,
    mode: zti_protocol::request::SearchMode,
    root: Option<String>,
    ctx: ClientCtx,
    tx: mpsc::Sender<AppMessage>,
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
            let _ = tx.send(AppMessage::SearchDone(results)).await;
        }
        Err(e) => {
            let _ = tx.send(AppMessage::SearchError(e.to_string())).await;
        }
    }
}

async fn do_remove_project(
    project_root: String,
    project_id: [u8; 32],
    ctx: ClientCtx,
    tx: mpsc::Sender<AppMessage>,
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
        let _ = tx.send(AppMessage::ProjectRemoveError(msg)).await;
        return;
    }

    let _ = tx.send(AppMessage::ProjectRemoved).await;
}

async fn do_index(
    project_root: String,
    refresh: bool,
    ctx: ClientCtx,
    tx: mpsc::Sender<AppMessage>,
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
                    refresh,
                    search_method: ctx.search_method.map(|m| m.as_str().to_string()),
                }),
                |frame| {
                    if let Response::IndexProgress(p) = frame {
                        let _ = tx_p.try_send(AppMessage::IndexProgress {
                            current: p.current,
                            total: p.total,
                            message: p.message,
                            is_reindex: refresh,
                        });
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
            let _ = tx.send(AppMessage::IndexComplete).await;
        }
        Err(e) => {
            let _ = tx.send(AppMessage::IndexError(e.to_string())).await;
        }
    }
}
