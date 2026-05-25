use crossterm::event::{self, KeyCode, KeyModifiers};

use super::app::{ActivePanel, App, DaemonStatus, Modal, Screen, SetupPhase};

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
    SetupNext,
    SetupPrev,
    SetupConfirm,
    SetupBack,
    SetupRetry,
    ChangeModel,
    OpenProjectDetail,
    DetailButtonNext,
    DetailButtonPrev,
    DetailConfirm,
    DetailBack,
    ConfirmRemoveYes,
    ConfirmRemoveNo,
}

pub fn map_key(key: &event::KeyEvent, app: &App) -> Action {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Action::Quit;
    }
    match &app.screen {
        Screen::Setup(phase) => map_setup_key(key, phase),
        Screen::Main => {
            if app.modal.is_some() {
                map_modal_key(key, app)
            } else {
                map_main_key(key, app)
            }
        }
    }
}

fn map_setup_key(key: &event::KeyEvent, phase: &SetupPhase) -> Action {
    match phase {
        SetupPhase::ModelSelection { .. } | SetupPhase::VariantSelection { .. } => match key.code {
            KeyCode::Char('j') | KeyCode::Down => Action::SetupNext,
            KeyCode::Char('k') | KeyCode::Up => Action::SetupPrev,
            KeyCode::Enter => Action::SetupConfirm,
            KeyCode::Char('q') | KeyCode::Esc => Action::SetupBack,
            _ => Action::None,
        },
        SetupPhase::Error { can_retry, .. } => match key.code {
            KeyCode::Char('r') if *can_retry => Action::SetupRetry,
            KeyCode::Char('q') | KeyCode::Esc => Action::Quit,
            _ => Action::None,
        },
        _ => match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Action::Quit,
            _ => Action::None,
        },
    }
}

fn map_main_key(key: &event::KeyEvent, app: &App) -> Action {
    match key.code {
        KeyCode::Char('q') if !in_search(app) => Action::Quit,
        KeyCode::Tab => Action::SwitchPanel,
        KeyCode::Char('/') if !in_search(app) => Action::FocusSearch,
        KeyCode::Enter if in_projects(app) => Action::OpenProjectDetail,
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
        KeyCode::Char('m') if !in_search(app) => Action::ChangeModel,
        KeyCode::Char(c) if in_search(app) => Action::Input(c),
        KeyCode::Backspace if in_search(app) => Action::Backspace,
        KeyCode::Esc if in_search(app) => Action::SwitchPanel,
        _ => Action::None,
    }
}

fn map_modal_key(key: &event::KeyEvent, app: &App) -> Action {
    match &app.modal {
        Some(Modal::ConfirmRemove) => match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => Action::ConfirmRemoveYes,
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => Action::ConfirmRemoveNo,
            _ => Action::None,
        },
        Some(Modal::Error { .. }) => match key.code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => Action::DetailBack,
            _ => Action::None,
        },
        Some(Modal::Reindexing { .. }) => Action::None,
        _ => match key.code {
            KeyCode::Tab | KeyCode::Char('l') | KeyCode::Right => Action::DetailButtonNext,
            KeyCode::BackTab | KeyCode::Char('h') | KeyCode::Left => Action::DetailButtonPrev,
            KeyCode::Enter => Action::DetailConfirm,
            KeyCode::Esc | KeyCode::Char('q') => Action::DetailBack,
            _ => Action::None,
        },
    }
}

fn in_search(app: &App) -> bool {
    matches!(app.active_panel, ActivePanel::Search)
}

fn in_projects(app: &App) -> bool {
    matches!(app.active_panel, ActivePanel::Projects)
}
