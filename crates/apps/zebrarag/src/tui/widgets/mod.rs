pub mod common;
pub mod modals;
pub mod projects;
pub mod results;
pub mod search;
pub mod setup;
pub mod status_bar;

use ratatui::Frame;

use super::app::{App, Screen};

pub fn draw(f: &mut Frame, app: &App, tick: u16) {
    match &app.screen {
        Screen::Setup(phase) => setup::draw(f, phase, tick),
        Screen::Main => draw_main(f, app, tick),
    }
}

fn draw_main(f: &mut Frame, app: &App, tick: u16) {
    let area = f.area();
    let chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(3),
            ratatui::layout::Constraint::Min(10),
            ratatui::layout::Constraint::Length(3),
        ])
        .split(area);

    status_bar::draw_status_bar(f, app, chunks[0]);
    draw_main_content(f, app, chunks[1]);
    draw_help_bar(f, app, chunks[2]);

    if app.modal.is_some() {
        modals::draw_modal(f, app, tick);
    }
}

fn draw_main_content(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let cols = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints([
            ratatui::layout::Constraint::Percentage(25),
            ratatui::layout::Constraint::Percentage(75),
        ])
        .split(area);

    projects::draw_projects(f, app, cols[0]);
    search::draw_search(f, app, cols[1]);
}

fn draw_help_bar(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    use ratatui::widgets::{Block, Borders, Paragraph};

    use super::app::{DaemonStatus, Modal};

    let keys = if app.modal.is_some() {
        match &app.modal {
            Some(Modal::ConfirmRemove) => "  y: confirm remove   n/Esc: cancel ",
            Some(Modal::Error { .. }) => "  Esc/Enter: dismiss ",
            Some(Modal::Indexing { .. }) => "  indexing in progress... ",
            Some(Modal::AddProject { .. }) => "  Enter: submit   Esc: cancel ",
            Some(Modal::ChangeIndexMethod { canonical_path, .. }) if canonical_path.is_some() => {
                "  j/k: navigate   Tab: switch   Enter: confirm   a: auto-recommend   Esc: back "
            }
            Some(Modal::ChangeIndexMethod { .. }) => {
                "  j/k: navigate   Enter: select   a: auto-recommend   Esc: back "
            }
            _ => "  Tab/\u{2190}\u{2192}: select   Enter: confirm   Esc: back ",
        }
    } else {
        match &app.daemon_status {
            DaemonStatus::Stopped => "  Tab: switch panel  r: restart daemon  q: quit ",
            DaemonStatus::Error(_) => {
                "  Tab: switch panel  r: restart daemon  m: change model  q: quit  (see daemon.log) "
            }
            _ => {
                "  Tab: panel  /: mode  Enter: search  j/k: scroll  PgUp/PgDn: page  s: stop  m: model  q: quit "
            }
        }
    };
    let block = Block::default().borders(Borders::ALL);
    let para = Paragraph::new(keys).block(block);
    f.render_widget(para, area);
}
