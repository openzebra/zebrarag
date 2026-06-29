use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem};

use super::super::app::{ActivePanel, App};

pub fn draw_projects(f: &mut Frame, app: &App, area: Rect) {
    let highlight = matches!(app.active_panel, ActivePanel::Projects);
    let border_style = if highlight {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };

    let mut items: Vec<ListItem> = Vec::with_capacity(app.projects.len() + 1);
    for (i, p) in app.projects.iter().enumerate() {
        let name = std::path::Path::new(&p.root_path)
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or(std::borrow::Cow::Borrowed("?"));
        let prefix = if i == app.selected_project {
            "> "
        } else {
            "  "
        };
        let style = if i == app.selected_project {
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Cyan)
        } else {
            Style::default()
        };
        let index_label = format!("#{} ", i + 1);
        let mut spans = Vec::with_capacity(5);
        spans.push(Span::styled(index_label, style));
        spans.push(Span::styled(prefix, style));
        spans.push(Span::styled(name, style));
        if !p.languages.is_empty() {
            let langs = format!("  ({})", p.languages.join(", "));
            spans.push(Span::styled(langs, Style::default().fg(Color::Gray)));
        }
        let line = Line::from(spans);
        items.push(ListItem::new(line));
    }

    let is_add_row = app.selected_project == app.projects.len();
    let add_style = if is_add_row {
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let add_prefix = if is_add_row { "> " } else { "  " };
    items.push(ListItem::new(Line::from(vec![
        Span::styled(add_prefix, add_style),
        Span::styled("[+] New", add_style),
    ])));

    let block = Block::default()
        .title(" Projects ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let list = List::new(items).block(block);
    f.render_widget(list, area);
}
