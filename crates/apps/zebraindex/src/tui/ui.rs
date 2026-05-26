use std::fmt::Write;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use zti_protocol::request::SearchMode;

use super::app::{ActivePanel, AddConfirmButton, App, DaemonStatus, DetailButton, Modal};

const PREVIEW_LINES: usize = 6;

const SPINNER: &[&str] = &[
    "\u{2807}", "\u{280b}", "\u{2819}", "\u{2838}",
    "\u{2830}", "\u{2826}", "\u{280e}", "\u{2803}",
];

pub fn spinner_ch(tick: u16) -> &'static str {
    SPINNER[tick as usize % SPINNER.len()]
}

pub fn centered_rect(pct_x: u16, pct_y: u16, area: Rect) -> Rect {
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - pct_y) / 2),
            Constraint::Percentage(pct_y),
            Constraint::Percentage((100 - pct_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - pct_x) / 2),
            Constraint::Percentage(pct_x),
            Constraint::Percentage((100 - pct_x) / 2),
        ])
        .split(vert[1])[1]
}

pub fn draw(f: &mut Frame, app: &App, tick: u16) {
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

    if app.modal.is_some() {
        draw_modal(f, app, tick);
    }
}

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let color = match &app.daemon_status {
        DaemonStatus::Running { .. } => Color::Green,
        DaemonStatus::Starting => Color::Yellow,
        DaemonStatus::Error(_) => Color::Red,
        _ => Color::DarkGray,
    };

    let indicator = match &app.daemon_status {
        DaemonStatus::Unknown => "?",
        DaemonStatus::Starting => "◌",
        DaemonStatus::Running { .. } => "●",
        DaemonStatus::Stopped => "○",
        DaemonStatus::Error(_) => "!",
    };

    let mut spans = Vec::with_capacity(5);
    spans.push(Span::styled("  Daemon: ", Style::default()));
    spans.push(Span::styled(indicator, Style::default().fg(color)));
    spans.push(Span::raw(" "));

    match &app.daemon_status {
        DaemonStatus::Unknown => spans.push(Span::raw("Checking...")),
        DaemonStatus::Starting => spans.push(Span::raw("Starting...")),
        DaemonStatus::Running {
            model_id,
            device,
            uptime_secs,
            loaded_models,
            loading_model,
        } => {
            let mins = uptime_secs / 60;
            let hrs = mins / 60;
            let mut text = String::with_capacity(128);
            write!(text, "Running  Model: {}  Device: {}", model_id, device).ok();
            if let Some(loading) = loading_model {
                write!(text, "  Loading: {}...", loading).ok();
            }
            write!(
                text,
                "  Models: {}  Uptime: {}h {}m",
                loaded_models.len(),
                hrs,
                mins % 60,
            )
            .ok();
            spans.push(Span::raw(text));
        }
        DaemonStatus::Stopped => spans.push(Span::raw("Stopped")),
        DaemonStatus::Error(e) => {
            let first_line = e.lines().next().unwrap_or(e.as_str());
            spans.push(Span::styled(
                format!("Error: {}", first_line),
                Style::default().fg(Color::Red),
            ));
        }
    }

    let line = Line::from(spans);
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
        let line = Line::from(vec![
            Span::styled(prefix, style),
            Span::styled(name, style),
        ]);
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

fn default_prefix(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Query => "query: ",
        SearchMode::Passage => "passage: ",
    }
}

fn draw_search(f: &mut Frame, app: &App, area: Rect) {
    let highlight = matches!(app.active_panel, ActivePanel::Search);

    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(5),
        ])
        .split(area);

    for (i, input) in app.search_inputs.iter().enumerate() {
        let is_active = highlight && i == app.active_input;
        let input_border = if is_active {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        };

        let prefix = match input.mode {
            SearchMode::Query => app.query_prefix.as_deref().unwrap_or(default_prefix(input.mode)),
            SearchMode::Passage => app.passage_prefix.as_deref().unwrap_or(default_prefix(input.mode)),
        };

        let input_block = Block::default()
            .title(prefix)
            .borders(Borders::ALL)
            .border_style(input_border);

        let input_para = if input.text.is_empty() && !is_active {
            Paragraph::new("  press Tab to switch")
                .block(input_block)
                .style(Style::default().fg(Color::DarkGray))
        } else {
            Paragraph::new(Line::from(vec![
                Span::raw("  "),
                Span::raw(input.text.as_str()),
            ]))
            .block(input_block)
        };
        f.render_widget(input_para, inner[i]);
    }

    let results_block = Block::default()
        .title(" Results ")
        .borders(Borders::ALL);

    if app.searching {
        let para = Paragraph::new("  searching...")
            .block(results_block)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(para, inner[2]);
    } else if let Some(ref err) = app.search_error {
        let para = Paragraph::new(format!("  error: {}", err))
            .block(results_block)
            .style(Style::default().fg(Color::Red));
        f.render_widget(para, inner[2]);
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

        let para = Paragraph::new(lines)
            .block(results_block)
            .wrap(Wrap { trim: false })
            .scroll((app.results_scroll, 0));
        f.render_widget(para, inner[2]);
    } else {
        let para = Paragraph::new("  no results yet")
            .block(results_block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(para, inner[2]);
    }
}

fn draw_help_bar(f: &mut Frame, app: &App, area: Rect) {
    let keys = if app.modal.is_some() {
        match &app.modal {
            Some(Modal::ConfirmRemove) => "  y: confirm remove   n/Esc: cancel ",
            Some(Modal::Error { .. }) => "  Esc/Enter: dismiss ",
            Some(Modal::Indexing { .. }) => "  indexing in progress... ",
            Some(Modal::AddProject { .. }) => "  Enter: submit   Esc: cancel ",
            Some(Modal::AddProjectConfirm { .. }) => {
                "  Tab/←→: select   Enter: confirm   Esc: cancel "
            }
            _ => "  Tab/←→: select   Enter: confirm   Esc: back ",
        }
    } else {
        match &app.daemon_status {
            DaemonStatus::Stopped => "  Tab: switch panel  r: restart daemon  q: quit ",
            DaemonStatus::Error(_) => {
                "  Tab: switch panel  r: restart daemon  m: change model  q: quit  (see daemon.log) "
            }
            _ => "  Tab: switch panel  /: search  Enter: open project  j/k: scroll  s: stop  m: change model  q: quit ",
        }
    };
    let block = Block::default().borders(Borders::ALL);
    let para = Paragraph::new(keys).block(block);
    f.render_widget(para, area);
}

fn draw_modal(f: &mut Frame, app: &App, tick: u16) {
    match &app.modal {
        Some(Modal::ProjectDetail { selected_button }) => {
            if let Some(p) = app.projects.get(app.selected_project) {
                draw_project_detail(f, p, *selected_button);
            }
        }
        Some(Modal::ConfirmRemove) => {
            if let Some(p) = app.projects.get(app.selected_project) {
                draw_confirm_remove(f, &p.root_path);
            }
        }
        Some(Modal::Error { message }) => {
            draw_modal_error(f, message);
        }
        Some(Modal::Indexing {
            current,
            total,
            message,
            is_reindex,
        }) => {
            draw_modal_indexing(f, tick, *current, *total, message, *is_reindex);
        }
        Some(Modal::AddProject { path_input, error }) => {
            draw_add_project(f, path_input, error.as_deref());
        }
        Some(Modal::AddProjectConfirm {
            canonical_path,
            already_indexed,
            selected_button,
        }) => {
            draw_add_project_confirm(f, canonical_path, *already_indexed, *selected_button);
        }
        None => {}
    }
}

fn draw_project_detail(f: &mut Frame, project: &zti_store::ProjectRow, selected: DetailButton) {
    let area = centered_rect(70, 60, f.area());
    f.render_widget(Clear, area);

    let name = std::path::Path::new(&project.root_path)
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or(std::borrow::Cow::Borrowed("?"));

    let title = format!(" Project: {} ", name);
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let ago_indexed = zti_common::format::format_elapsed(project.last_indexed_ns);
    let ago_created = zti_common::format::format_elapsed(project.created_at_ns);
    let search = project
        .search_method
        .as_deref()
        .unwrap_or("unknown");
    let langs = if project.languages.is_empty() {
        String::from("unknown")
    } else {
        project.languages.join(", ")
    };

    let dim_str = format!("{} (dim={})", project.model_id, project.model_dim);

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Root:          "),
            Span::styled(&project.root_path, Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("  Model:         "),
            Span::styled(dim_str, Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("  Languages:     "),
            Span::styled(langs, Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("  Chunks:        "),
            Span::styled(
                project.total_chunks.to_string(),
                Style::default().fg(Color::Gray),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Files:         "),
            Span::styled(
                project.total_files.to_string(),
                Style::default().fg(Color::Gray),
            ),
        ]),
        Line::from(vec![
            Span::raw("  Search method: "),
            Span::styled(search, Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("  Last indexed:  "),
            Span::styled(ago_indexed, Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("  Created:       "),
            Span::styled(ago_created, Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  ─────────────────────────────────────",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        draw_buttons(selected),
    ];

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, area);
}

fn render_button_row(buttons: &[(&str, bool)]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(buttons.len() * 3);
    for (i, (label, is_sel)) in buttons.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("    "));
        }
        let style = if *is_sel {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let prefix = if *is_sel { "> [" } else { "  [" };
        spans.push(Span::styled(
            format!("{}{}{}", prefix, label, if *is_sel { "] <" } else { "]" }),
            style,
        ));
    }
    Line::from(spans)
}

fn draw_buttons(selected: DetailButton) -> Line<'static> {
    render_button_row(&[
        ("Remove", selected == DetailButton::Remove),
        ("Reindex", selected == DetailButton::Reindex),
        ("Back", selected == DetailButton::Back),
    ])
}

fn draw_confirm_remove(f: &mut Frame, root_path: &str) {
    let area = centered_rect(50, 25, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" Confirm Remove ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  This will permanently delete all indexed",
            Style::default().fg(Color::Yellow),
        )),
        Line::from(Span::styled(
            "  data for this project. This cannot be undone.",
            Style::default().fg(Color::Yellow),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Project: "),
            Span::styled(root_path, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  y: remove   n/Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];

    let para = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_modal_error(f: &mut Frame, message: &str) {
    let area = centered_rect(50, 25, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" Error ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", message),
            Style::default().fg(Color::Red),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Esc/Enter: dismiss",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];

    let para = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_modal_indexing(
    f: &mut Frame,
    tick: u16,
    current: u64,
    total: u64,
    message: &str,
    is_reindex: bool,
) {
    let area = centered_rect(55, 30, f.area());
    f.render_widget(Clear, area);

    let title = if is_reindex { " Reindexing " } else { " Indexing " };
    let label = if is_reindex {
        "Reindexing project..."
    } else {
        "Indexing project..."
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let bar_width: usize = 30;
    let filled = if total > 0 {
        ((current as f64 / total as f64) * bar_width as f64) as usize
    } else {
        0
    };
    let bar: String = format!(
        "[{}{}] {}/{}",
        "=".repeat(filled),
        " ".repeat(bar_width - filled),
        current,
        total,
    );

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("  {} ", spinner_ch(tick)),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(label),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(bar, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::raw(message),
        ]),
        Line::from(""),
    ];

    let para = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_add_project(f: &mut Frame, path_input: &str, error: Option<&str>) {
    let area = centered_rect(55, 30, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" Add Project ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let mut lines = Vec::with_capacity(8);
    lines.push(Line::from(""));
    lines.push(Line::from(Span::raw(
        "  Enter the path to the project directory:",
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(path_input, Style::default().fg(Color::White)),
        Span::styled("\u{258f}", Style::default().fg(Color::Gray)),
    ]));
    lines.push(Line::from(""));

    if let Some(err) = error {
        lines.push(Line::from(Span::styled(
            format!("  \u{2717} {}", err),
            Style::default().fg(Color::Red),
        )));
        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "  Enter: submit   Esc: cancel",
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));

    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_add_project_confirm(
    f: &mut Frame,
    canonical_path: &str,
    already_indexed: bool,
    selected: AddConfirmButton,
) {
    let area = centered_rect(55, 30, f.area());
    f.render_widget(Clear, area);

    let name = std::path::Path::new(canonical_path)
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or(std::borrow::Cow::Borrowed("?"));

    let status = if already_indexed {
        "Already indexed (will re-index)"
    } else {
        "Not indexed"
    };

    let block = Block::default()
        .title(" Confirm Index ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Project:   "),
            Span::styled(name, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("  Path:      "),
            Span::styled(canonical_path, Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("  Status:    "),
            Span::styled(status, Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  ─────────────────────────────────────",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        render_button_row(&[
            ("Confirm", selected == AddConfirmButton::Confirm),
            ("Cancel", selected == AddConfirmButton::Cancel),
        ]),
    ];

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, area);
}
