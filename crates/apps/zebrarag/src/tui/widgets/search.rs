use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use super::super::app::{ActivePanel, App};
use super::results::draw_results;

pub fn draw_search(f: &mut Frame, app: &App, area: Rect) {
    let highlight = matches!(app.active_panel, ActivePanel::Search);

    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(5)])
        .split(area);

    draw_search_input(f, app, inner[0], highlight);
    draw_results(f, app, inner[1]);
}

fn draw_search_input(f: &mut Frame, app: &App, area: Rect, highlight: bool) {
    let input_border = if highlight {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };

    let prefix = match app.search_input.mode {
        zrag_protocol::request::SearchMode::Query => {
            app.query_prefix.as_deref().unwrap_or("query: ")
        }
        zrag_protocol::request::SearchMode::Passage => {
            app.passage_prefix.as_deref().unwrap_or("passage: ")
        }
    };

    let input_block = Block::default()
        .title(prefix)
        .borders(Borders::ALL)
        .border_style(input_border);

    if app.search_input.text.is_empty() && !highlight {
        let para = Paragraph::new("  press Tab to start")
            .block(input_block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(para, area);
    } else {
        let para = Paragraph::new(Line::from(vec![
            Span::raw("  "),
            Span::raw(app.search_input.text.as_str()),
        ]))
        .block(input_block);
        f.render_widget(para, area);
    }
}
