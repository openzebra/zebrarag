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

const REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/hicaru/zebra_tree_indexer/refs/heads/master/models.toml";

pub fn run_tui(
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
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
            ..App::default()
        };

        let (tx, mut rx) = mpsc::channel::<AppMessage>(32);

        if model.is_some() {
            app.screen = app::Screen::Main;
            spawn_daemon_monitor(&app, &tx);
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
                variant: Some(cfg.default_variant),
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
        let _ = config::save(&p.model_id, "auto");
        let _ = tx
            .send(AppMessage::ConfigResolved {
                model: Some(p.model_id),
                variant: Some("auto".into()),
            })
            .await;
        return;
    }

    let _ = tx
        .send(AppMessage::ConfigResolved {
            model: None,
            variant: None,
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
    let result = tokio::task::spawn_blocking(move || {
        let hw = zti_hw::probe();
        let variant = zti_embed::OnnxVariant::Auto;
        zti_embed::model_registry::resolve_model_files(&id, &variant, &hw)
    })
    .await;

    match result {
        Ok(Ok(_)) => {
            let _ = tx.send(AppMessage::ModelDownloaded(model_id)).await;
        }
        Ok(Err(e)) => {
            let _ = tx
                .send(AppMessage::ModelDownloadError(e.to_string()))
                .await;
        }
        Err(e) => {
            let _ = tx
                .send(AppMessage::ModelDownloadError(e.to_string()))
                .await;
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
            variant,
        } => {
            app.model = Some(Arc::from(m.as_str()));
            app.variant = variant.map(|v| Arc::from(v.as_str()));
            app.should_run.store(true, Ordering::Relaxed);
            app.screen = app::Screen::Main;
            spawn_daemon_monitor(app, tx);
        }
        AppMessage::RegistryLoaded(mut entries) => {
            let hw = zti_hw::probe();
            registry::sort_by_hardware(&mut entries, &hw.device);
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
            if let Some(ref reg) = app.setup_registry
                && let Some(entry) = reg.iter().find(|e| e.model_id.as_str() == model_id.as_ref())
            {
                let variants = entry.variant_list();
                app.screen = app::Screen::Setup(app::SetupPhase::VariantSelection {
                    model_id,
                    variants,
                    selected: 0,
                });
                return;
            }
            let _ = tx
                .send(AppMessage::SetupComplete {
                    model: model_id,
                    variant: Arc::from("auto"),
                })
                .await;
        }
        AppMessage::ModelDownloadError(msg) => {
            app.screen = app::Screen::Setup(app::SetupPhase::Error {
                message: msg,
                can_retry: false,
            });
        }
        AppMessage::SetupComplete { model, variant } => {
            app.model = Some(Arc::clone(&model));
            app.variant = Some(Arc::clone(&variant));
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

fn spawn_daemon_monitor(app: &App, tx: &mpsc::Sender<AppMessage>) {
    let ctx = ClientCtx::from_app(app);
    let tx_m = tx.clone();
    let should_run = app.should_run.clone();
    tokio::spawn(async move {
        daemon_monitor(tx_m, ctx, should_run).await;
    });
}

async fn handle_action(app: &mut App, action: event::Action, tx: &mpsc::Sender<AppMessage>) {
    match action {
        event::Action::Quit => app.should_quit = true,

        event::Action::SetupNext => match &mut app.screen {
            app::Screen::Setup(app::SetupPhase::ModelSelection { entries, selected })
                if *selected + 1 < entries.len() =>
            {
                *selected += 1;
            }
            app::Screen::Setup(app::SetupPhase::VariantSelection {
                variants, selected, ..
            }) if *selected + 1 < variants.len() => {
                *selected += 1;
            }
            _ => {}
        },
        event::Action::SetupPrev => {
            if let app::Screen::Setup(
                app::SetupPhase::ModelSelection { selected, .. }
                | app::SetupPhase::VariantSelection { selected, .. },
            ) = &mut app.screen
                && *selected > 0
            {
                *selected -= 1;
            }
        }
        event::Action::SetupConfirm => match &app.screen {
            app::Screen::Setup(app::SetupPhase::ModelSelection { entries, selected }) => {
                let entry = &entries[*selected];
                let model_id: Arc<str> = Arc::from(entry.model_id.as_str());
                if entry.is_downloaded() {
                    let variants = entry.variant_list();
                    app.screen = app::Screen::Setup(app::SetupPhase::VariantSelection {
                        model_id,
                        variants,
                        selected: 0,
                    });
                } else {
                    let id = Arc::clone(&model_id);
                    app.screen = app::Screen::Setup(app::SetupPhase::DownloadingModel { model_id });
                    let tx_c = tx.clone();
                    tokio::spawn(async move { download_model(id, tx_c).await });
                }
            }
            app::Screen::Setup(app::SetupPhase::VariantSelection {
                model_id,
                variants,
                selected,
            }) => {
                let variant_str: Arc<str> = if *selected == 0 {
                    Arc::from("auto")
                } else {
                    Arc::clone(&variants[*selected].0)
                };
                let save_model = Arc::clone(model_id);
                let save_variant = Arc::clone(&variant_str);
                let launch_model = Arc::clone(model_id);
                let launch_variant = Arc::clone(&variant_str);
                let complete_model = Arc::clone(model_id);

                if let Err(e) = config::save(&save_model, &save_variant) {
                    app.screen = app::Screen::Setup(app::SetupPhase::Error {
                        message: format!("Failed to save config: {e}"),
                        can_retry: false,
                    });
                    return;
                }

                app.screen = app::Screen::Setup(app::SetupPhase::Launching {
                    model_id: launch_model,
                    variant: launch_variant,
                });

                let _ = tx
                    .send(AppMessage::SetupComplete {
                        model: complete_model,
                        variant: variant_str,
                    })
                    .await;
            }
            _ => {}
        },
        event::Action::SetupBack => match &app.screen {
            app::Screen::Setup(app::SetupPhase::ModelSelection { .. }) => {
                if app.model.is_some() {
                    app.should_run.store(true, Ordering::Relaxed);
                    app.screen = app::Screen::Main;
                } else {
                    app.should_quit = true;
                }
            }
            app::Screen::Setup(app::SetupPhase::VariantSelection { .. }) => {
                if let Some(ref entries) = app.setup_registry {
                    app.screen = app::Screen::Setup(app::SetupPhase::ModelSelection {
                        entries: Arc::clone(entries),
                        selected: 0,
                    });
                }
            }
            _ => {}
        },
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
                app.search_input.push(c);
            }
        }
        event::Action::Backspace => {
            if let Some(app::Modal::AddProject {
                ref mut path_input, ..
            }) = app.modal
            {
                path_input.pop();
            } else {
                app.search_input.pop();
            }
        }
        event::Action::SubmitSearch => {
            if !app.search_input.is_empty() && !app.searching {
                app.searching = true;
                app.search_error = None;
                let query = app.search_input.clone();
                let root = app.selected_project_root().map(|s| s.to_string());
                let ctx = ClientCtx::from_app(app);
                let tx_clone = tx.clone();
                tokio::spawn(async move {
                    do_search(query, root, ctx, tx_clone).await;
                });
            }
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
        event::Action::DetailButtonNext => {
            match &mut app.modal {
                Some(app::Modal::ProjectDetail { selected_button }) => {
                    *selected_button = match selected_button {
                        app::DetailButton::Remove => app::DetailButton::Reindex,
                        app::DetailButton::Reindex => app::DetailButton::Back,
                        app::DetailButton::Back => app::DetailButton::Remove,
                    };
                }
                Some(app::Modal::AddProjectConfirm { selected_button, .. }) => {
                    *selected_button = match selected_button {
                        app::AddConfirmButton::Confirm => app::AddConfirmButton::Cancel,
                        app::AddConfirmButton::Cancel => app::AddConfirmButton::Confirm,
                    };
                }
                _ => {}
            }
        }
        event::Action::DetailButtonPrev => {
            match &mut app.modal {
                Some(app::Modal::ProjectDetail { selected_button }) => {
                    *selected_button = match selected_button {
                        app::DetailButton::Remove => app::DetailButton::Back,
                        app::DetailButton::Reindex => app::DetailButton::Remove,
                        app::DetailButton::Back => app::DetailButton::Reindex,
                    };
                }
                Some(app::Modal::AddProjectConfirm { selected_button, .. }) => {
                    *selected_button = match selected_button {
                        app::AddConfirmButton::Confirm => app::AddConfirmButton::Cancel,
                        app::AddConfirmButton::Cancel => app::AddConfirmButton::Confirm,
                    };
                }
                _ => {}
            }
        }
        event::Action::DetailConfirm => {
            match app.modal.take() {
                Some(app::Modal::ProjectDetail { selected_button }) => {
                    match selected_button {
                        app::DetailButton::Remove => {
                            app.modal = Some(app::Modal::ConfirmRemove);
                        }
                        app::DetailButton::Reindex => {
                            if let Some(root) = app.selected_project_root_owned() {
                                let ctx = ClientCtx::from_app(app);
                                let tx_c = tx.clone();
                                tokio::spawn(async move {
                                    do_index(root, true, ctx, tx_c).await;
                                });
                            }
                        }
                        app::DetailButton::Back => {}
                    }
                }
                Some(app::Modal::AddProjectConfirm {
                    canonical_path,
                    selected_button,
                    ..
                }) => match selected_button {
                    app::AddConfirmButton::Confirm => {
                        app.modal = Some(app::Modal::Indexing {
                            current: 0,
                            total: 0,
                            message: String::with_capacity(64),
                            is_reindex: false,
                        });
                        let ctx = ClientCtx::from_app(app);
                        let tx_c = tx.clone();
                        tokio::spawn(async move {
                            do_index(canonical_path, false, ctx, tx_c).await;
                        });
                    }
                    app::AddConfirmButton::Cancel => {
                        app.modal = Some(app::Modal::AddProject {
                            path_input: canonical_path,
                            error: None,
                        });
                    }
                },
                other => {
                    app.modal = other;
                }
            }
        }
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
            if let Some(app::Modal::AddProject {
                ref path_input, ..
            }) = app.modal
            {
                let trimmed = path_input.trim();
                if trimmed.is_empty() {
                    if let Some(app::Modal::AddProject {
                        ref mut error, ..
                    }) = app.modal
                    {
                        *error = Some(String::from("Path cannot be empty"));
                    }
                } else {
                    let path = std::path::Path::new(trimmed);
                    if !path.is_dir() {
                        if let Some(app::Modal::AddProject {
                            ref mut error, ..
                        }) = app.modal
                        {
                            *error = Some(String::from("Directory does not exist"));
                        }
                    } else if let Ok(canonical) = path.canonicalize() {
                        let canonical_str = canonical.to_string_lossy().into_owned();
                        let already_indexed = app
                            .projects
                            .iter()
                            .any(|p| p.root_path == canonical_str);
                        app.modal = Some(app::Modal::AddProjectConfirm {
                            canonical_path: canonical_str,
                            already_indexed,
                            selected_button: app::AddConfirmButton::default(),
                        });
                    } else if let Some(app::Modal::AddProject {
                        ref mut error, ..
                    }) = app.modal
                    {
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
    variant: Option<Arc<str>>,
    query_prefix: Option<Arc<str>>,
    passage_prefix: Option<Arc<str>>,
}

impl ClientCtx {
    fn from_app(app: &App) -> Self {
        Self {
            client: app.client.clone(),
            model: app.model.clone(),
            variant: app.variant.clone(),
            query_prefix: app.query_prefix.clone(),
            passage_prefix: app.passage_prefix.clone(),
        }
    }

    fn deref_opts(&self) -> (Option<&str>, Option<&str>, Option<&str>, Option<&str>) {
        (
            self.model.as_deref(),
            self.variant.as_deref(),
            self.query_prefix.as_deref(),
            self.passage_prefix.as_deref(),
        )
    }
}

async fn ensure_client(
    client: &Arc<Mutex<Option<zti_ipc_client::Client>>>,
    model: Option<&str>,
    variant: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
) -> anyhow::Result<()> {
    let mut guard = client.lock().await;
    if guard.is_none() {
        let mut c = zti_ipc_client::Client::connect(
            Duration::from_secs(10),
            model,
            variant,
            query_prefix,
            passage_prefix,
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

async fn try_connect(
    ctx: &ClientCtx,
    tx: &mpsc::Sender<AppMessage>,
) {
    let (m, v, qp, pp) = ctx.deref_opts();
    if let Err(e) = ensure_client(&ctx.client, m, v, qp, pp).await {
        let mut msg = e.to_string();
        read_daemon_log_tail(&mut msg);
        let _ = tx
            .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Error(
                msg,
            )))
            .await;
    }
}

async fn daemon_monitor(
    tx: mpsc::Sender<AppMessage>,
    ctx: ClientCtx,
    should_run: Arc<AtomicBool>,
) {
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
    root: Option<String>,
    ctx: ClientCtx,
    tx: mpsc::Sender<AppMessage>,
) {
    let result = async {
        let (m, v, qp, pp) = ctx.deref_opts();
        ensure_client(&ctx.client, m, v, qp, pp).await?;

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
                mode: zti_protocol::request::SearchMode::default(),
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
        let (m, v, qp, pp) = ctx.deref_opts();
        ensure_client(&ctx.client, m, v, qp, pp).await?;

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
        let (m, v, qp, pp) = ctx.deref_opts();
        ensure_client(&ctx.client, m, v, qp, pp).await?;

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
