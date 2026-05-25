use crossterm::event::{self, KeyCode, KeyModifiers};

use super::app::{ActivePanel, App, DaemonStatus};

pub enum Action {
    Quit,
    SwitchPanel,
    FocusSearch,
    SubmitSearch,
    ScrollUp,
    ScrollDown,
    SelectPrevProject,
    SelectNextProject,
    StopDaemon,
    RestartDaemon,
    Input(char),
    Backspace,
    None,
}

pub fn map_key(key: &event::KeyEvent, app: &App) -> Action {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Action::Quit;
    }
    match key.code {
        KeyCode::Char('q') if !in_search(app) => Action::Quit,
        KeyCode::Tab => Action::SwitchPanel,
        KeyCode::Char('/') if !in_search(app) => Action::FocusSearch,
        KeyCode::Enter => Action::SubmitSearch,
        KeyCode::Char('j') | KeyCode::Down if in_projects(app) => Action::SelectNextProject,
        KeyCode::Char('k') | KeyCode::Up if in_projects(app) => Action::SelectPrevProject,
        KeyCode::Char('j') | KeyCode::Down if !in_search(app) => Action::ScrollDown,
        KeyCode::Char('k') | KeyCode::Up if !in_search(app) => Action::ScrollUp,
        KeyCode::Char('s') if !in_search(app) => Action::StopDaemon,
        KeyCode::Char('r')
            if !in_search(app) && matches!(app.daemon_status, DaemonStatus::Stopped) =>
        {
            Action::RestartDaemon
        }
        KeyCode::Char(c) if in_search(app) => Action::Input(c),
        KeyCode::Backspace if in_search(app) => Action::Backspace,
        KeyCode::Esc if in_search(app) => Action::SwitchPanel,
        _ => Action::None,
    }
}

fn in_search(app: &App) -> bool {
    matches!(app.active_panel, ActivePanel::Search)
}

fn in_projects(app: &App) -> bool {
    matches!(app.active_panel, ActivePanel::Projects)
}
