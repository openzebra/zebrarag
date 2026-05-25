use std::sync::Arc;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};

use super::app::SetupPhase;
use super::registry::ModelEntry;

const SPINNER: &[&str] = &[
    "\u{2807}", "\u{280b}", "\u{2819}", "\u{2838}",
    "\u{2830}", "\u{2826}", "\u{280e}", "\u{2803}",
];

fn centered_rect(pct_x: u16, pct_y: u16, area: Rect) -> Rect {
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

fn spinner_ch(tick: u16) -> &'static str {
    SPINNER[tick as usize % SPINNER.len()]
}

pub fn draw(f: &mut Frame, phase: &SetupPhase, tick: u16) {
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Black)),
        f.area(),
    );
    match phase {
        SetupPhase::Resolving => draw_spinner(f, "Checking configuration...", tick),
        SetupPhase::FetchingRegistry => draw_spinner(f, "Downloading model catalog...", tick),
        SetupPhase::ModelSelection { entries, selected } => {
            draw_model_selection(f, entries, *selected);
        }
        SetupPhase::DownloadingModel { model_id } => draw_download(f, model_id, tick),
        SetupPhase::VariantSelection {
            model_id,
            variants,
            selected,
        } => draw_variant_selection(f, model_id, variants, *selected),
        SetupPhase::Launching {
            model_id, variant, ..
        } => draw_launching(f, model_id, variant, tick),
        SetupPhase::Error {
            message, can_retry, ..
        } => draw_error(f, message, *can_retry),
    }
}

fn draw_spinner(f: &mut Frame, message: &str, tick: u16) {
    let area = centered_rect(50, 20, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" zebraindex setup ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("  {} ", spinner_ch(tick)),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(message),
        ]),
        Line::from(""),
    ];
    let para = Paragraph::new(text).block(block);
    f.render_widget(para, area);
}

fn draw_download(f: &mut Frame, model_id: &str, tick: u16) {
    let area = centered_rect(55, 25, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" zebraindex setup ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Model: "),
            Span::styled(model_id, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("  {} ", spinner_ch(tick)),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw("Downloading from HuggingFace..."),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Please wait, this may take a few minutes.",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
    ];
    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_launching(f: &mut Frame, model_id: &str, variant: &str, tick: u16) {
    let area = centered_rect(50, 20, f.area());
    f.render_widget(Clear, area);

    let block = Block::default()
        .title(" zebraindex setup ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("  {} ", spinner_ch(tick)),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw("Starting daemon..."),
        ]),
        Line::from(vec![
            Span::raw("  Model: "),
            Span::styled(model_id, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::raw("  Variant: "),
            Span::styled(variant, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(""),
    ];
    let para = Paragraph::new(text).block(block);
    f.render_widget(para, area);
}

fn draw_model_selection(f: &mut Frame, entries: &[ModelEntry], selected: usize) {
    let area = centered_rect(80, 80, f.area());
    f.render_widget(Clear, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(area);

    let mut items: Vec<ListItem> = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "> " } else { "  " };
        let style = if is_sel {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let tag = if entry.is_downloaded() {
            Span::styled(" [downloaded]", Style::default().fg(Color::Green))
        } else {
            Span::raw("")
        };

        let line1 = Line::from(vec![
            Span::styled(prefix, style),
            Span::styled(&entry.model_id, style),
            Span::styled(
                format!("  ({})", entry.parameters),
                Style::default().fg(Color::DarkGray),
            ),
            tag,
        ]);

        let line2 = Line::from(vec![
            Span::raw("    "),
            Span::styled(&entry.description, Style::default().fg(Color::Gray)),
        ]);

        let techs = entry.technologies.join(", ");
        let line3 = Line::from(vec![
            Span::raw("    "),
            Span::styled(techs, Style::default().fg(Color::DarkGray)),
        ]);

        items.push(ListItem::new(vec![line1, line2, line3]));
    }

    let block = Block::default()
        .title(" Select Embedding Model ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    f.render_widget(List::new(items).block(block), layout[0]);

    let help = Paragraph::new("  j/k: navigate   Enter: select   q: quit")
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, layout[1]);
}

fn draw_variant_selection(
    f: &mut Frame,
    model_id: &str,
    variants: &[(Arc<str>, Arc<str>)],
    selected: usize,
) {
    let area = centered_rect(70, 60, f.area());
    f.render_widget(Clear, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(3)])
        .split(area);

    let mut items: Vec<ListItem> = Vec::with_capacity(variants.len());
    for (i, (name, desc)) in variants.iter().enumerate() {
        let is_sel = i == selected;
        let prefix = if is_sel { "> " } else { "  " };
        let style = if is_sel {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let line1 = Line::from(vec![
            Span::styled(prefix, style),
            Span::styled(name.as_ref(), style),
        ]);
        let line2 = Line::from(vec![
            Span::raw("    "),
            Span::styled(desc.as_ref(), Style::default().fg(Color::DarkGray)),
        ]);
        items.push(ListItem::new(vec![line1, line2]));
    }

    let title = format!(" Select ONNX Variant: {} ", model_id);
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    f.render_widget(List::new(items).block(block), layout[0]);

    let help = Paragraph::new("  j/k: navigate   Enter: select   Esc: back")
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(help, layout[1]);
}

fn draw_error(f: &mut Frame, message: &str, can_retry: bool) {
    let area = centered_rect(50, 25, f.area());
    f.render_widget(Clear, area);

    let keys = if can_retry {
        "  r: retry   q: quit"
    } else {
        "  q: quit"
    };

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
        Line::from(Span::styled(keys, Style::default().fg(Color::DarkGray))),
        Line::from(""),
    ];

    let para = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}
