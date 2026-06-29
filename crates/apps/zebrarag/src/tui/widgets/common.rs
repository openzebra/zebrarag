use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

pub const SPINNER: &[&str] = &[
    "\u{2807}", "\u{280b}", "\u{2819}", "\u{2838}", "\u{2830}", "\u{2826}", "\u{280e}", "\u{2803}",
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

pub fn render_button_row(buttons: &[(&'static str, bool)]) -> Line<'static> {
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
        let (prefix, suffix) = if *is_sel {
            ("> [", "] <")
        } else {
            ("  [", "]")
        };
        spans.push(Span::styled(prefix, style));
        spans.push(Span::styled(*label, style));
        spans.push(Span::styled(suffix, style));
    }
    Line::from(spans)
}

pub const FILLED_20: &str = "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}";
pub const EMPTY_20: &str = "\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}";

pub fn bar_slice(full: &'static str, n: usize) -> &'static str {
    let byte_len = full.len() / 20 * n;
    &full[..byte_len]
}

pub fn render_bar(label: &'static str, pct: u8) -> Line<'static> {
    let filled = (pct as usize).saturating_mul(20) / 100;
    let empty = 20 - filled;
    let color = match pct {
        90..=100 => Color::Green,
        60..=89 => Color::Cyan,
        30..=59 => Color::Yellow,
        _ => Color::Red,
    };
    Line::from(vec![
        Span::styled(
            format!("  {:<14}", label),
            Style::default().fg(Color::White),
        ),
        Span::styled(bar_slice(FILLED_20, filled), Style::default().fg(color)),
        Span::styled(
            bar_slice(EMPTY_20, empty),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(format!("  {:>3}%", pct), Style::default().fg(Color::White)),
    ])
}
