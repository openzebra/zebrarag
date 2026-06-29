use std::borrow::Cow;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::super::app::{App, PREVIEW_LINES};

pub fn draw_results(f: &mut Frame, app: &App, area: Rect) {
    app.results_visible_height
        .set(area.height.saturating_sub(2) as usize);

    let results_block = Block::default().title(" Results ").borders(Borders::ALL);

    if app.searching {
        let para = Paragraph::new("  searching...")
            .block(results_block)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(para, area);
    } else if let Some(ref err) = app.search_error {
        let para = Paragraph::new(format!("  error: {}", err))
            .block(results_block)
            .style(Style::default().fg(Color::Red));
        f.render_widget(para, area);
    } else if let Some(ref results) = app.search_results {
        let total_hits = results.hits.len();
        let vis = app.results_visible_height.get();
        let page_info: Cow<'_, str> = if app.results_total_lines > 0 && vis > 0 {
            let current_page = (app.results_scroll / vis.max(1)) + 1;
            let total_pages = (app.results_total_lines + vis - 1) / vis.max(1);
            Cow::Owned(format!(" [{}/{}]", current_page, total_pages))
        } else {
            Cow::Borrowed("")
        };

        let header = format!(
            "\u{2500}\u{2500} Results ({} hits) \u{2500}\u{2500}{}",
            total_hits, page_info
        );
        let header_line = Line::from(Span::styled(header, Style::default().fg(Color::DarkGray)));

        let mut lines: Vec<Line> = Vec::with_capacity(1 + app.results_total_lines);
        lines.push(header_line);

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
                    Span::styled("  \u{250a} ", Style::default().fg(Color::DarkGray)),
                    Span::raw(*line),
                ]));
            }
            if all_lines.len() > PREVIEW_LINES {
                lines.push(Line::from(Span::styled(
                    "  \u{250a} ...",
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }

        let para = Paragraph::new(lines)
            .block(results_block)
            .wrap(Wrap { trim: false })
            .scroll((app.results_scroll as u16, 0));
        f.render_widget(para, area);
    } else {
        let para = Paragraph::new("  no results yet")
            .block(results_block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(para, area);
    }
}
