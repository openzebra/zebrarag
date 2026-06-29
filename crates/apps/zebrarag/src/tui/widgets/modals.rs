use ratatui::Frame;

use std::borrow::Cow;

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use super::super::app::{App, DetailButton, Modal};
use super::common::{EMPTY_20, FILLED_20, bar_slice, centered_rect, render_button_row, spinner_ch};
use super::setup::draw_method_selection_modal;

pub fn draw_modal(f: &mut Frame, app: &App, tick: u16) {
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
        Some(m @ Modal::Indexing { .. }) => {
            draw_modal_indexing(f, tick, m, app);
        }
        Some(Modal::AddProject { path_input, error }) => {
            draw_add_project(f, path_input, error.as_deref());
        }
        Some(Modal::ChangeIndexMethod {
            methods,
            selected,
            canonical_path,
            already_indexed,
            selected_button,
            ..
        }) => {
            draw_method_selection_modal(
                f,
                methods,
                *selected,
                canonical_path.as_deref(),
                already_indexed.unwrap_or(false),
                canonical_path.is_some(),
                *selected_button,
            );
        }
        None => {}
    }
}

fn draw_project_detail(f: &mut Frame, project: &zrag_store::ProjectRow, selected: DetailButton) {
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

    let ago_indexed = zrag_common::format::format_elapsed(project.last_indexed_ns);
    let ago_created = zrag_common::format::format_elapsed(project.created_at_ns);
    let search = project.search_method.as_deref().unwrap_or("unknown");
    let langs: Cow<'_, str> = if project.languages.is_empty() {
        Cow::Borrowed("unknown")
    } else {
        Cow::Owned(project.languages.join(", "))
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
            Span::styled(langs.as_ref(), Style::default().fg(Color::Gray)),
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
            "  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        draw_buttons(selected),
        Line::from(Span::styled(
            "  Enter: select • f: full reindex",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, area);
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

    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
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

    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

#[allow(clippy::too_many_arguments)]
fn phase_row(
    pdef: zrag_protocol::response::IndexPhase,
    name: &'static str,
    active_order: u8,
    current: u64,
    total: u64,
    files: u64,
    chunks: u64,
    message: &str,
) -> Line<'static> {
    use std::cmp::Ordering;
    let ord = pdef.order();
    let (icon, color) = match ord.cmp(&active_order) {
        Ordering::Less => ("\u{2713}", Color::Green),
        Ordering::Equal => ("\u{25b6}", Color::Cyan),
        Ordering::Greater => ("\u{00b7}", Color::DarkGray),
    };
    let head = Span::styled(format!("  {} {}  ", icon, name), Style::default().fg(color));
    if ord < active_order {
        let suffix = match pdef {
            zrag_protocol::response::IndexPhase::Dsl => {
                format!("{} files parsed", files)
            }
            zrag_protocol::response::IndexPhase::Gather => {
                format!("{} chunks", chunks)
            }
            zrag_protocol::response::IndexPhase::Tokenize => {
                format!("{} chunks", chunks)
            }
            _ => String::new(),
        };
        Line::from(vec![
            head,
            Span::styled(suffix, Style::default().fg(Color::Gray)),
        ])
    } else if ord == active_order {
        let bar = if total > 0 {
            let filled = ((current as f64 / total as f64) * 20.0) as usize;
            format!(
                "[{}{}]",
                bar_slice(FILLED_20, filled.min(20)),
                bar_slice(EMPTY_20, 20 - filled.min(20)),
            )
        } else {
            String::from("[                    ]")
        };
        let tail = if total > 0 {
            Span::styled(
                format!(" {}/{}", current, total),
                Style::default().fg(Color::White),
            )
        } else {
            Span::styled(message.to_string(), Style::default().fg(Color::Gray))
        };
        Line::from(vec![
            head,
            Span::styled(bar, Style::default().fg(Color::Cyan)),
            tail,
        ])
    } else {
        Line::from(vec![
            head,
            Span::styled("pending", Style::default().fg(Color::DarkGray)),
        ])
    }
}

fn draw_modal_indexing(f: &mut Frame, tick: u16, m: &super::super::app::Modal, app: &App) {
    let (phase, current, total, message, is_reindex, started_at, files, chunks) = match m {
        super::super::app::Modal::Indexing {
            phase,
            current,
            total,
            message,
            is_reindex,
            started_at,
            files,
            chunks,
            ..
        } => (
            phase,
            *current,
            *total,
            message.as_str(),
            *is_reindex,
            *started_at,
            *files,
            *chunks,
        ),
        _ => return,
    };

    let area = centered_rect(55, 46, f.area());
    f.render_widget(Clear, area);

    let title = if is_reindex {
        " Reindexing "
    } else {
        " Indexing "
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let elapsed = started_at.elapsed();
    let elapsed_str = format!(
        "elapsed {}:{:02}",
        elapsed.as_secs() / 60,
        elapsed.as_secs() % 60
    );
    let label = if is_reindex {
        "Reindexing project..."
    } else {
        "Indexing project..."
    };

    let active_order = phase.order();

    let (device, _cpus, _mem_mb) = app.effective_hardware();
    let model = app.model.as_deref().unwrap_or("--");

    let phase_labels: &[(zrag_protocol::response::IndexPhase, &str)] = &[
        (zrag_protocol::response::IndexPhase::Dsl, "Parse"),
        (zrag_protocol::response::IndexPhase::Gather, "Gather"),
        (zrag_protocol::response::IndexPhase::Tokenize, "Tokenize"),
        (zrag_protocol::response::IndexPhase::Embed, "Embed"),
        (zrag_protocol::response::IndexPhase::BuildIndex, "Index"),
        (zrag_protocol::response::IndexPhase::Finish, "Finish"),
    ];

    let mut lines = Vec::with_capacity(16);
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            format!("  {} ", spinner_ch(tick)),
            Style::default().fg(Color::Yellow),
        ),
        Span::raw(label),
        Span::raw("  "),
        Span::styled(elapsed_str, Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::from(""));

    for (pdef, name) in phase_labels {
        lines.push(phase_row(
            *pdef,
            name,
            active_order,
            current,
            total,
            files,
            chunks,
            message,
        ));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  \u{2500}\u{2500} Hardware \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(vec![
        Span::styled("  Device: ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{:15}", device)),
        Span::styled("Model: ", Style::default().fg(Color::DarkGray)),
        Span::raw(model),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Esc/c: cancel",
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));

    let para = Paragraph::new(lines)
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
