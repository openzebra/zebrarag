//! Heading detection over a page's rendered lines.
//!
//! Pure data analysis — no I/O, no PDF parsing. The body font size is the mode
//! of the rendered sizes; any line materially larger than the body (and short
//! enough to be a title) is treated as a heading.

use crate::interpreter::Line;

/// A line is considered a heading when its font size is at least this multiple
/// of the page's body size. 1.2× separates most chapter/section titles from
/// body copy without flagging slightly-larger lead paragraphs.
pub const HEADING_RATIO: f32 = 1.2;

/// Maximum characters in a detected heading; longer runs are body text even at
/// a heading size.
pub const HEADING_MAX_LEN: usize = 120;

/// Mode of the rendered font sizes, rounded to one decimal so near-equal
/// sizes (10.0 vs 10.02) collapse together. Returns 0.0 for an empty page or a
/// page with no positive sizes.
#[must_use]
pub fn body_font_size(lines: &[Line]) -> f32 {
    if lines.is_empty() {
        return 0.0;
    }
    let mut sizes: Vec<f32> = lines
        .iter()
        .map(|l| (l.font_size * 10.0).round() / 10.0)
        .filter(|&s| s > 0.0)
        .collect();
    if sizes.is_empty() {
        return 0.0;
    }
    sizes.sort_by(|a, b| a.total_cmp(b));
    let mut best = sizes[0];
    let mut best_count: u32 = 1;
    let mut cur = sizes[0];
    let mut cur_count: u32 = 1;
    for &s in &sizes[1..] {
        if (s - cur).abs() < 0.01 {
            cur_count += 1;
        } else {
            if cur_count > best_count {
                best_count = cur_count;
                best = cur;
            }
            cur = s;
            cur_count = 1;
        }
    }
    if cur_count > best_count {
        best = cur;
    }
    best
}

/// Minimum letter ratio for a heading candidate. Lines dominated by digits
/// and punctuation (exercise markers like `x70.[M33]`) are noise even when
/// they render at a heading size.
pub const HEADING_MIN_LETTER_RATIO: f32 = 0.5;

/// Pick the first heading-sized short line as the page heading, given the
/// precomputed `body_size`. Returns `None` when the page has uniform sizing,
/// every candidate is empty/too long, or the candidate is mostly non-letter
/// characters (digit/punctuation noise).
#[must_use]
pub fn pick_heading(lines: &[Line], body_size: f32) -> Option<String> {
    if body_size <= 0.0 {
        return None;
    }
    for line in lines {
        if line.font_size < body_size * HEADING_RATIO {
            continue;
        }
        let trimmed = line.text.trim();
        if trimmed.is_empty() || trimmed.len() > HEADING_MAX_LEN {
            continue;
        }
        let letters = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        if letters as f32 / (trimmed.len() as f32) < HEADING_MIN_LETTER_RATIO {
            continue;
        }
        return Some(trimmed.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, size: f32) -> Line {
        Line {
            text: text.into(),
            font_size: size,
        }
    }

    #[test]
    fn body_font_size_mode_of_uniform() {
        let lines = [line("a", 10.0), line("b", 10.0), line("c", 10.0)];
        assert_eq!(body_font_size(&lines), 10.0);
    }

    #[test]
    fn body_font_size_ignores_rare_headings() {
        let lines = [
            line("Title", 18.0),
            line("body", 10.0),
            line("body", 10.0),
            line("body", 10.0),
        ];
        assert_eq!(body_font_size(&lines), 10.0);
    }

    #[test]
    fn body_font_size_empty_is_zero() {
        assert_eq!(body_font_size(&[]), 0.0);
    }

    #[test]
    fn pick_heading_detects_first_oversized_line() {
        let lines = [
            line("Chapter 1: Intro", 18.0),
            line("Welcome to the book.", 10.0),
            line("More body text.", 10.0),
        ];
        assert_eq!(
            pick_heading(&lines, 10.0).as_deref(),
            Some("Chapter 1: Intro")
        );
    }

    #[test]
    fn pick_heading_rejects_overlong() {
        let long = "x".repeat(HEADING_MAX_LEN + 50);
        let lines = [line(&long, 24.0)];
        assert!(pick_heading(&lines, 10.0).is_none());
    }

    #[test]
    fn pick_heading_none_when_uniform_size() {
        let lines = [line("line one", 10.0), line("line two", 10.0)];
        assert!(pick_heading(&lines, 10.0).is_none());
    }

    #[test]
    fn pick_heading_none_with_zero_body_size() {
        let lines = [line("x", 18.0)];
        assert!(pick_heading(&lines, 0.0).is_none());
    }
}
