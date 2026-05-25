use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

use super::app::{ActivePanel, App, DaemonStatus};

const PREVIEW_LINES: usize = 6;

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(f.area());

    draw_status_bar(f, app, chunks[0]);
    draw_main(f, app, chunks[1]);
    draw_help_bar(f, app, chunks[2]);
}

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let (indicator, spans) = match &app.daemon_status {
        DaemonStatus::Unknown => ("?", vec![Span::raw("Checking...")]),
        DaemonStatus::Starting => ("◌", vec![Span::raw("Starting...")]),
        DaemonStatus::Running {
            model_id,
            device,
            uptime_secs,
        } => {
            let mins = uptime_secs / 60;
            let hrs = mins / 60;
            (
                "●",
                vec![Span::raw(format!(
                    "Running  Model: {}  Device: {}  Uptime: {}h {}m",
                    model_id, device, hrs, mins % 60
                ))],
            )
        }
        DaemonStatus::Stopped => ("○", vec![Span::raw("Stopped")]),
        DaemonStatus::Error(e) => ("!", vec![Span::raw(format!("Error: {}", e))]),
    };

    let color = match &app.daemon_status {
        DaemonStatus::Running { .. } => Color::Green,
        DaemonStatus::Starting => Color::Yellow,
        DaemonStatus::Error(_) => Color::Red,
        _ => Color::DarkGray,
    };

    let line = Line::from(vec![
        Span::styled("  Daemon: ", Style::default()),
        Span::styled(indicator, Style::default().fg(color)),
        Span::raw(" "),
    ]
    .into_iter()
    .chain(spans)
    .collect::<Vec<_>>());

    let block = Block::default()
        .title(" zebraindex ")
        .borders(Borders::ALL);
    let para = Paragraph::new(line).block(block);
    f.render_widget(para, area);
}

fn draw_main(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
        .split(area);

    draw_projects(f, app, cols[0]);
    draw_search(f, app, cols[1]);
}

fn draw_projects(f: &mut Frame, app: &App, area: Rect) {
    let highlight = matches!(app.active_panel, ActivePanel::Projects);
    let border_style = if highlight {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };

    let items: Vec<ListItem> = app
        .projects
        .iter()
        .enumerate()
        .map(|(i, p)| {
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
            let line = Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(name, style),
            ]);
            ListItem::new(line)
        })
        .collect();

    let block = Block::default()
        .title(" Projects ")
        .borders(Borders::ALL)
        .border_style(border_style);

    if app.projects.is_empty() {
        let placeholder = Paragraph::new("  (no projects)")
            .block(block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(placeholder, area);
    } else {
        let list = List::new(items).block(block);
        f.render_widget(list, area);
    }
}

fn draw_search(f: &mut Frame, app: &App, area: Rect) {
    let highlight = matches!(app.active_panel, ActivePanel::Search);
    let border_style = if highlight {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    };

    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(5)])
        .split(area);

    let input_block = Block::default()
        .title(" Search ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let input_text = if app.search_input.is_empty() && !highlight {
        "  press / to search"
    } else {
        ""
    };
    let input_para = if app.search_input.is_empty() && !highlight {
        Paragraph::new(input_text)
            .block(input_block)
            .style(Style::default().fg(Color::DarkGray))
    } else {
        Paragraph::new(format!("  {}", app.search_input)).block(input_block)
    };
    f.render_widget(input_para, inner[0]);

    let results_block = Block::default()
        .title(" Results ")
        .borders(Borders::ALL);

    if app.searching {
        let para = Paragraph::new("  searching...")
            .block(results_block)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(para, inner[1]);
    } else if let Some(ref err) = app.search_error {
        let para = Paragraph::new(format!("  error: {}", err))
            .block(results_block)
            .style(Style::default().fg(Color::Red));
        f.render_widget(para, inner[1]);
    } else if let Some(ref results) = app.search_results {
        let mut lines: Vec<Line> = Vec::with_capacity(1 + results.hits.len() * (2 + PREVIEW_LINES));
        let header = format!("── Results ({} hits) ──", results.hits.len());
        lines.push(Line::from(Span::styled(
            header,
            Style::default().fg(Color::DarkGray),
        )));

        for (i, hit) in results.hits.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("#{} [{:.4}] ", i + 1, hit.score),
                    Style::default().fg(Color::Green),
                ),
                Span::raw(format!(
                    "{}:{}-{}",
                    hit.file_path, hit.start_line, hit.end_line
                )),
            ]));

            let all_lines: Vec<&str> = hit.content.lines().collect();
            let visible = all_lines.len().min(PREVIEW_LINES);
            for line in &all_lines[..visible] {
                lines.push(Line::from(vec![
                    Span::styled("  ┊ ", Style::default().fg(Color::DarkGray)),
                    Span::raw(*line),
                ]));
            }
            if all_lines.len() > PREVIEW_LINES {
                lines.push(Line::from(Span::styled(
                    "  ┊ ...",
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }

        let scroll = app.results_scroll as usize;
        let visible: Vec<Line> = lines.into_iter().skip(scroll).collect();
        let para = Paragraph::new(visible)
            .block(results_block)
            .wrap(Wrap { trim: false });
        f.render_widget(para, inner[1]);
    } else {
        let para = Paragraph::new("  no results yet")
            .block(results_block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(para, inner[1]);
    }
}

fn draw_help_bar(f: &mut Frame, app: &App, area: Rect) {
    let keys = match &app.daemon_status {
        DaemonStatus::Stopped => "  Tab: switch panel  r: restart daemon  q: quit ",
        _ => "  Tab: switch panel  /: search  Enter: submit  j/k: scroll  s: stop  q: quit ",
    };
    let block = Block::default().borders(Borders::ALL);
    let para = Paragraph::new(keys).block(block);
    f.render_widget(para, area);
}
