mod app;
mod event;
mod ui;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use tokio::sync::{Mutex, mpsc};
use zti_protocol::request::Request;
use zti_protocol::response::Response;

use app::{App, AppMessage};

pub fn run_tui(
    model: Option<&str>,
    _variant: Option<zti_embed::OnnxVariant>,
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

        let mut app = App::new();
        let (tx, mut rx) = mpsc::channel::<AppMessage>(32);

        let client = app.client.clone();
        let tx_monitor = tx.clone();
        let model_owned = model.map(|s| s.to_string());
        let query_prefix_owned = query_prefix.map(|s| s.to_string());
        let passage_prefix_owned = passage_prefix.map(|s| s.to_string());
        tokio::spawn(async move {
            daemon_monitor(
                tx_monitor,
                client,
                model_owned.as_deref(),
                query_prefix_owned.as_deref(),
                passage_prefix_owned.as_deref(),
            )
            .await;
        });

        loop {
            terminal.draw(|f| ui::draw(f, &app))?;

            while let Ok(msg) = rx.try_recv() {
                app.apply_message(msg);
            }

            if crossterm::event::poll(Duration::from_millis(50))?
                && let crossterm::event::Event::Key(key) = crossterm::event::read()?
            {
                let action = event::map_key(&key, &app);
                handle_action(&mut app, action, &tx).await;
            }

            if app.should_quit {
                break;
            }
        }

        terminal::disable_raw_mode()?;
        execute!(std::io::stdout(), LeaveAlternateScreen)?;
        Ok(())
    })
}

async fn ensure_client(
    client: &Arc<Mutex<Option<zti_ipc_client::Client>>>,
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
) -> anyhow::Result<()> {
    let mut guard = client.lock().await;
    if guard.is_none() {
        let mut c = zti_ipc_client::Client::connect(
            Duration::from_secs(10),
            model,
            None,
            query_prefix,
            passage_prefix,
        )
        .await?;
        c.handshake().await?;
        *guard = Some(c);
    }
    Ok(())
}

async fn daemon_monitor(
    tx: mpsc::Sender<AppMessage>,
    client: Arc<Mutex<Option<zti_ipc_client::Client>>>,
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
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
            let _ = tx
                .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Stopped))
                .await;
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        }

        let status = {
            let mut guard = client.lock().await;
            match guard.as_mut() {
                Some(c) => match c.request(Request::DaemonStatus).await {
                    Ok(Response::DaemonStatus(info)) => Some(app::DaemonStatus::Running {
                        model_id: info.model_id,
                        device: info.device,
                        uptime_secs: info.uptime_secs,
                    }),
                    Ok(_) => None,
                    Err(_) => {
                        *guard = None;
                        None
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
            if ensure_client(&client, model, query_prefix, passage_prefix)
                .await
                .is_err()
            {
                let _ = tx
                    .send(AppMessage::DaemonStatusUpdate(app::DaemonStatus::Stopped))
                    .await;
            }
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
    client: Arc<Mutex<Option<zti_ipc_client::Client>>>,
    tx: mpsc::Sender<AppMessage>,
) {
    let result = async {
        let mut guard = client.lock().await;
        if guard.is_none() {
            let mut c = zti_ipc_client::Client::connect(
                Duration::from_secs(10),
                None,
                None,
                None,
                None,
            )
            .await?;
            c.handshake().await?;
            *guard = Some(c);
        }
        let c = guard.as_mut().unwrap();

        let project_root = match root {
            Some(r) => r,
            None => {
                let projects = zti_store::list_projects().await?;
                if projects.is_empty() {
                    anyhow::bail!("No indexed projects");
                }
                projects[0].root_path.clone()
            }
        };

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

async fn handle_action(app: &mut App, action: event::Action, tx: &mpsc::Sender<AppMessage>) {
    match action {
        event::Action::Quit => app.should_quit = true,
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
            if app.selected_project + 1 < app.projects.len() {
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
            app.search_input.push(c);
        }
        event::Action::Backspace => {
            app.search_input.pop();
        }
        event::Action::SubmitSearch => {
            if !app.search_input.is_empty() && !app.searching {
                app.searching = true;
                app.search_error = None;
                let query = app.search_input.clone();
                let root = app.selected_project_root().map(|s| s.to_string());
                let tx_clone = tx.clone();
                let client = app.client.clone();
                tokio::spawn(async move {
                    do_search(query, root, client, tx_clone).await;
                });
            }
        }
        event::Action::StopDaemon => {
            let client = app.client.clone();
            tokio::spawn(async move {
                let mut guard = client.lock().await;
                if let Some(c) = guard.take() {
                    let mut c = c;
                    let _ = c.request(Request::Stop).await;
                }
            });
        }
        event::Action::RestartDaemon => {
            {
                let mut guard = app.client.lock().await;
                *guard = None;
            }
            app.daemon_status = app::DaemonStatus::Starting;
        }
        event::Action::None => {}
    }
}
