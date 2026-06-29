use std::borrow::Cow;
use std::fmt::Write;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders};

use super::super::app::{App, DaemonStatus};
use super::super::registry::RemoteProvider;

pub fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let color = match &app.daemon_status {
        DaemonStatus::Running { .. } => Color::Green,
        DaemonStatus::Starting => Color::Yellow,
        DaemonStatus::Error(_) => Color::Red,
        _ => Color::DarkGray,
    };

    let indicator = match &app.daemon_status {
        DaemonStatus::Unknown => "?",
        DaemonStatus::Starting => "\u{25CC}",
        DaemonStatus::Running { .. } => "\u{25CF}",
        DaemonStatus::Stopped => "\u{25CB}",
        DaemonStatus::Error(_) => "!",
    };

    let status_label = match &app.daemon_status {
        DaemonStatus::Unknown => "Unknown",
        DaemonStatus::Starting => "Starting",
        DaemonStatus::Running { .. } => "Running",
        DaemonStatus::Stopped => "Stopped",
        DaemonStatus::Error(_) => "Error",
    };

    let model_label: Cow<'_, str> = match app.model.as_deref() {
        None => Cow::Borrowed("--"),
        Some(model) => match RemoteProvider::from_model_id(model) {
            Some((provider, remote)) => Cow::Owned(format!("{}:{remote}", provider.as_str())),
            None => Cow::Borrowed(model),
        },
    };
    let model_style = if app
        .model
        .as_deref()
        .is_some_and(|model| RemoteProvider::from_model_id(model).is_some())
    {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(color)
    };
    let dtype = app.model_dtype.as_deref().unwrap_or("--");

    let uptime_str = match &app.daemon_status {
        DaemonStatus::Running { uptime_secs, .. } => {
            let mins = uptime_secs / 60;
            let hrs = mins / 60;
            if hrs > 0 {
                Some(format!("{}h {}m", hrs, mins % 60))
            } else {
                Some(format!("{}m", mins))
            }
        }
        _ => None,
    };

    let (device, cpus, mem_total_mb) = app.effective_hardware();

    let block = Block::default().title(" zebrarag ").borders(Borders::ALL);
    if let DaemonStatus::Error(err_msg) = &app.daemon_status {
        let first_line = err_msg.lines().next().unwrap_or(err_msg.as_str());
        let mut text = String::with_capacity(160);
        write!(
            text,
            "{}  Error: {}  (see daemon.log)",
            indicator, first_line
        )
        .ok();
        let line = Line::from(vec![Span::styled(text, Style::default().fg(color))]);
        f.render_widget(ratatui::widgets::Paragraph::new(line).block(block), area);
        return;
    }

    let mut tail = String::with_capacity(96);
    write!(tail, "  DType: {dtype}  Device: {device}  CPU: {cpus}").ok();
    if mem_total_mb > 0 {
        write!(tail, "  RAM: {mem_total_mb}M").ok();
    }
    if let Some(uptime) = &uptime_str {
        write!(tail, "  {uptime}").ok();
    }

    let line = Line::from(vec![
        Span::styled(
            format!("{indicator}  {status_label}  Model: "),
            Style::default().fg(color),
        ),
        Span::styled(model_label, model_style),
        Span::styled(tail, Style::default().fg(color)),
    ]);
    f.render_widget(ratatui::widgets::Paragraph::new(line).block(block), area);
}
