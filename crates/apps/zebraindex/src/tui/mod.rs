mod actions;
mod app;
mod config;
mod event;
mod registry;
mod tasks;
mod widgets;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use app::{App, AppMessage, DaemonStatus, Modal, Screen, SetupPhase};
use tasks::{
    fetch_registry, resolve_startup, spawn_daemon_monitor, spawn_refresh_projects,
};

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
            local_hardware: Some(zti_hw::probe()),
            ..App::default()
        };

        let (tx, mut rx) = mpsc::channel::<AppMessage>(32);

        if model.is_some() {
            app.screen = Screen::Main;
            spawn_daemon_monitor(&mut app, &tx);
        } else {
            let tx_r = tx.clone();
            tokio::spawn(async move { resolve_startup(tx_r).await });
        }

        let mut tick: u16 = 0;

        loop {
            terminal.draw(|f| widgets::draw(f, &app, tick))?;

            while let Ok(msg) = rx.try_recv() {
                dispatch(&mut app, msg, &tx).await;
            }

            if crossterm::event::poll(Duration::from_millis(50))?
                && let crossterm::event::Event::Key(key) = crossterm::event::read()?
            {
                let action = event::map_key(&key, &app);
                actions::handle_action(&mut app, action, &tx).await;
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

async fn dispatch(app: &mut App, msg: AppMessage, tx: &mpsc::Sender<AppMessage>) {
    match msg {
        AppMessage::ConfigResolved { model: None, .. } => {
            app.screen = Screen::Setup(SetupPhase::FetchingRegistry);
            let tx_c = tx.clone();
            tokio::spawn(async move { fetch_registry(tx_c).await });
        }
        AppMessage::ConfigResolved {
            model: Some(m),
            search_method,
            model_dtype,
        } => {
            if let Some(dt) = &model_dtype {
                let _ = config::save(&m, search_method.as_deref(), Some(dt));
            }
            app.model = Some(Arc::from(m.as_str()));
            if let Some(s) = &search_method {
                app.search_method = zti_ann::SearchMethod::parse(s);
            }
            if let Some(d) = model_dtype {
                app.model_dtype = Some(Arc::from(d));
            }
            app.should_run.store(true, Ordering::Relaxed);
            app.screen = Screen::Main;
            spawn_daemon_monitor(app, tx);
        }
        AppMessage::RegistryLoaded(mut entries) => {
            registry::sort_by_hardware(&mut entries);
            let shared: Arc<[registry::ModelEntry]> = entries.into();
            app.setup_registry = Some(Arc::clone(&shared));
            app.screen = Screen::Setup(SetupPhase::ModelSelection {
                entries: shared,
                selected: 0,
            });
        }
        AppMessage::RegistryError(msg) => {
            app.screen = Screen::Setup(SetupPhase::Error {
                message: msg,
                can_retry: true,
            });
        }
        AppMessage::ModelDownloaded(model_id) => {
            let _ = tx
                .send(AppMessage::SetupComplete { model: model_id })
                .await;
        }
        AppMessage::ModelDownloadError(msg) => {
            app.screen = Screen::Setup(SetupPhase::Error {
                message: msg,
                can_retry: false,
            });
        }
        AppMessage::SetupComplete { model } => {
            app.model = Some(Arc::clone(&model));
            let client = app.client.clone();
            tokio::spawn(async move {
                let mut guard = client.lock().await;
                if let Some(mut c) = guard.take() {
                    let _ = c.request(zti_protocol::request::Request::Stop).await;
                }
            });
            app.should_run.store(true, Ordering::Relaxed);
            app.screen = Screen::Main;
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
            app.modal = Some(Modal::Error { message: e });
        }
        AppMessage::IndexComplete => {
            app.modal = None;
            spawn_refresh_projects(tx);
        }
        AppMessage::IndexProgress {
            phase,
            current,
            total,
            message,
            is_reindex,
        } => {
            let (started_at, project_root, mut files, mut chunks) = match &app.modal {
                Some(Modal::Indexing { started_at, project_root, files, chunks, .. }) =>
                    (*started_at, project_root.clone(), *files, *chunks),
                _ => (std::time::Instant::now(), String::new(), 0, 0),
            };
            if phase == zti_protocol::response::IndexPhase::Gather {
                files = total;
            }
            if phase == zti_protocol::response::IndexPhase::Tokenize {
                chunks = total;
            }
            app.modal = Some(Modal::Indexing {
                project_root,
                phase,
                current,
                total,
                message,
                is_reindex,
                started_at,
                files,
                chunks,
            });
        }
        AppMessage::IndexCancelled => {
            app.modal = None;
        }
        AppMessage::IndexError(e) => {
            app.modal = Some(Modal::Error { message: e });
        }
        AppMessage::DaemonEnvLoaded {
            cpus: env_cpus,
            mem_total_mb: env_mem,
        } => {
            if let DaemonStatus::Running {
                ref mut cpus,
                ref mut mem_total_mb,
                ..
            } = app.daemon_status
            {
                *cpus = env_cpus;
                *mem_total_mb = env_mem;
            }
        }
        other => app.apply_message(other),
    }
}
