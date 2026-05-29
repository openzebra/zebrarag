use ratatui::Frame;

use std::borrow::Cow;

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use super::common::{centered_rect, render_button_row, spinner_ch};
use super::super::app::{
    App, DetailButton, Modal,
};
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

fn draw_modal_indexing(
    f: &mut Frame,
    tick: u16,
    m: &super::super::app::Modal,
    app: &App,
) {
    let (phase, current, total, message, is_reindex, started_at) = match m {
        super::super::app::Modal::Indexing {
            phase,
            current,
            total,
            message,
            is_reindex,
            started_at, ..
        } => (phase, *current, *total, message.as_str(), *is_reindex, *started_at),
        _ => return,
    };

    let area = centered_rect(55, 40, f.area());
    f.render_widget(Clear, area);

    let title = if is_reindex { " Reindexing " } else { " Indexing " };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let bar_width: usize = 30;
    let pct = if total > 0 {
        (current as f64 / total as f64 * 100.0) as u8
    } else {
        0
    };
    let filled = if total > 0 {
        ((current as f64 / total as f64) * bar_width as f64) as usize
    } else {
        0
    };

    let label = if is_reindex {
        "Reindexing project..."
    } else {
        "Indexing project..."
    };

    let eta: Cow<'_, str> = if current > 0 && total > 0 {
        let elapsed = started_at.elapsed().as_secs_f64();
        let rate = current as f64 / elapsed;
        let remaining = (total - current) as f64 / rate;
        if remaining.is_finite() && remaining > 0.0 {
            Cow::Owned(format!("  ETA: ~{:.0}s", remaining))
        } else {
            Cow::Borrowed("")
        }
    } else {
        Cow::Borrowed("")
    };

    let (device, cpus, mem_mb) = app.effective_hardware();
    let model = app.model.as_deref().unwrap_or("--");
    let dtype = app.model_dtype.as_deref().unwrap_or("--");

    let is_gpu = matches!(device.to_ascii_lowercase().as_str(), "metal" | "cuda");
    let mem_label = if is_gpu { "VRAM:" } else { "RAM:" };

    let ram_str = if mem_mb > 0 {
        Cow::Owned(format!("{} MB", mem_mb))
    } else {
        Cow::Borrowed("--")
    };

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
            Span::styled(
                format!(
                    "[{}{}] {}/{}  {}%{}",
                    "\u{2588}".repeat(filled),
                    "\u{2591}".repeat(bar_width - filled),
                    current,
                    total,
                    pct,
                    eta,
                ),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Phase:   ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{phase}")),
        ]),
        Line::from(vec![
            Span::styled("  Message: ", Style::default().fg(Color::DarkGray)),
            Span::raw(message),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  \u{2500}\u{2500} Hardware \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(vec![
            Span::styled("  Device: ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{:15}", device)),
            Span::styled("CPU cores: ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{}", cpus)),
        ]),
        Line::from(vec![
            Span::styled(format!("  {}  ", mem_label), Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{:<16}", ram_str)),
            Span::styled("Model:  ", Style::default().fg(Color::DarkGray)),
            Span::raw(model),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("DType: ", Style::default().fg(Color::DarkGray)),
            Span::raw(dtype),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Esc/c: cancel",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];

    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
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


