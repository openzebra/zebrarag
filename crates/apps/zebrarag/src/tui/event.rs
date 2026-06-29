use crossterm::event::{self, KeyCode, KeyModifiers};

use super::app::{ActivePanel, App, DaemonStatus, Modal, Screen, SetupPhase};

pub enum Action {
    Quit,
    SwitchPanel,
    FocusSearch,
    ToggleSearchMode,
    SubmitSearch,
    ScrollUp,
    ScrollDown,
    PageUp,
    PageDown,
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
    DetailForceReindex,
    ConfirmRemoveYes,
    ConfirmRemoveNo,
    SubmitPath,
    SetupAutoRecommend,
    CancelIndex,
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
        SetupPhase::ModelSelection { .. }
        | SetupPhase::DTypeSelection { .. }
        | SetupPhase::RemoteModelSelection { .. } => match key.code {
            KeyCode::Char('j') | KeyCode::Down => Action::SetupNext,
            KeyCode::Char('k') | KeyCode::Up => Action::SetupPrev,
            KeyCode::Enter => Action::SetupConfirm,
            KeyCode::Char('q') | KeyCode::Esc => Action::SetupBack,
            _ => Action::None,
        },
        SetupPhase::IndexMethodSelection { .. } => match key.code {
            KeyCode::Char('j') | KeyCode::Down => Action::SetupNext,
            KeyCode::Char('k') | KeyCode::Up => Action::SetupPrev,
            KeyCode::Enter => Action::SetupConfirm,
            KeyCode::Char('a') => Action::SetupAutoRecommend,
            KeyCode::Char('q') | KeyCode::Esc => Action::SetupBack,
            _ => Action::None,
        },
        SetupPhase::ApiKeyEntry { .. } => match key.code {
            KeyCode::Char(c) => Action::Input(c),
            KeyCode::Backspace => Action::Backspace,
            KeyCode::Enter => Action::SetupConfirm,
            KeyCode::Esc => Action::SetupBack,
            _ => Action::None,
        },
        SetupPhase::FetchingRemoteModels { .. } => match key.code {
            KeyCode::Esc => Action::SetupBack,
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
        KeyCode::Char('/') if in_search(app) && app.search_input.text.is_empty() => {
            Action::ToggleSearchMode
        }
        KeyCode::Enter if in_projects(app) => Action::OpenProjectDetail,
        KeyCode::Enter => Action::SubmitSearch,
        KeyCode::Down if !in_search(app) => Action::SelectNextProject,
        KeyCode::Up if !in_search(app) => Action::SelectPrevProject,
        KeyCode::Char('j') if in_search(app) && app.search_input.text.is_empty() => {
            Action::ScrollDown
        }
        KeyCode::Char('k') if in_search(app) && app.search_input.text.is_empty() => {
            Action::ScrollUp
        }
        KeyCode::Char('j') if in_projects(app) => Action::SelectNextProject,
        KeyCode::Char('k') if in_projects(app) => Action::SelectPrevProject,
        KeyCode::PageDown => Action::PageDown,
        KeyCode::PageUp => Action::PageUp,
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
        Some(Modal::Indexing { .. }) => match key.code {
            KeyCode::Esc | KeyCode::Char('c') => Action::CancelIndex,
            _ => Action::None,
        },
        Some(Modal::AddProject { .. }) => match key.code {
            KeyCode::Enter => Action::SubmitPath,
            KeyCode::Esc => Action::DetailBack,
            KeyCode::Char(c) => Action::Input(c),
            KeyCode::Backspace => Action::Backspace,
            _ => Action::None,
        },
        Some(Modal::ChangeIndexMethod { canonical_path, .. }) => {
            let is_add = canonical_path.is_some();
            match key.code {
                KeyCode::Char('j') | KeyCode::Down => Action::SetupNext,
                KeyCode::Char('k') | KeyCode::Up => Action::SetupPrev,
                KeyCode::Tab | KeyCode::Char('l') | KeyCode::Right if is_add => {
                    Action::DetailButtonNext
                }
                KeyCode::BackTab | KeyCode::Char('h') | KeyCode::Left if is_add => {
                    Action::DetailButtonPrev
                }
                KeyCode::Enter => Action::DetailConfirm,
                KeyCode::Char('a') => Action::SetupAutoRecommend,
                KeyCode::Esc | KeyCode::Char('q') => Action::DetailBack,
                _ => Action::None,
            }
        }
        Some(Modal::ProjectDetail { .. }) => match key.code {
            KeyCode::Tab | KeyCode::Char('l') | KeyCode::Right => Action::DetailButtonNext,
            KeyCode::BackTab | KeyCode::Char('h') | KeyCode::Left => Action::DetailButtonPrev,
            KeyCode::Char('f') => Action::DetailForceReindex,
            KeyCode::Enter => Action::DetailConfirm,
            KeyCode::Esc | KeyCode::Char('q') => Action::DetailBack,
            _ => Action::None,
        },
        _ => Action::None,
    }
}

fn in_search(app: &App) -> bool {
    matches!(app.active_panel, ActivePanel::Search)
}

fn in_projects(app: &App) -> bool {
    matches!(app.active_panel, ActivePanel::Projects)
}
