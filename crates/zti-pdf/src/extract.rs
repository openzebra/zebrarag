//! Per-page extraction orchestrator: load the document with `lopdf`, walk its
//! pages, interpret each page's content stream, and detect a heading. All
//! heavy lifting lives in [`crate::interpreter`] (text) and [`crate::heading`]
//! (analysis); this module is wiring only.

use anyhow::{Result, anyhow};
use lopdf::Document;

use crate::heading::{body_font_size, pick_heading};
use crate::interpreter::interpret;

/// Plain text plus an optional detected heading for one PDF page. `page` is
/// 1-based to match PDF page labelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageText {
    pub page: u32,
    pub text: String,
    pub heading: Option<String>,
}

/// Extract per-page text and detected headings from a PDF byte stream.
///
/// Returns one entry per page in ascending page order. Pages whose content
/// stream cannot be parsed yield an empty `text` and `None` heading rather
/// than aborting the whole document; callers should warn-and-skip pages or
/// the whole file as appropriate when every page comes back empty.
pub fn extract_pages(bytes: &[u8]) -> Result<Vec<PageText>> {
    let doc = Document::load_mem(bytes).map_err(|e| anyhow!("pdf load failed: {e}"))?;
    // `get_pages` returns page-number (1-based) → object id in ascending order.
    let pages = doc.get_pages();
    let mut out = Vec::with_capacity(pages.len());
    for (page_num, page_id) in pages {
        let content = match doc.get_page_content(page_id) {
            Ok(c) => c,
            Err(_) => {
                // Page object with no decodable content stream — emit an empty
                // entry so the caller still sees the page boundary.
                out.push(PageText {
                    page: page_num,
                    text: String::new(),
                    heading: None,
                });
                continue;
            }
        };
        let lines = interpret(&content);
        let (text, heading) = assemble_page(&lines);
        out.push(PageText {
            page: page_num,
            text,
            heading,
        });
    }
    Ok(out)
}

/// Concatenate rendered lines into page text and pick the page heading.
fn assemble_page(lines: &[crate::interpreter::Line]) -> (String, Option<String>) {
    let total: usize = lines.iter().map(|l| l.text.len() + 1).sum();
    let mut text = String::with_capacity(total);
    let body_size = body_font_size(lines);
    let heading = pick_heading(lines, body_size);
    for line in lines {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&line.text);
    }
    (text, heading)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Line;

    #[test]
    fn assemble_page_concatenates_lines() {
        let lines = [
            Line {
                text: "a".into(),
                font_size: 10.0,
            },
            Line {
                text: "b".into(),
                font_size: 10.0,
            },
        ];
        let (text, heading) = assemble_page(&lines);
        assert_eq!(text, "a\nb");
        assert!(heading.is_none());
    }

    #[test]
    fn assemble_page_empty_lines_produce_empty_text() {
        let lines: Vec<Line> = vec![];
        let (text, heading) = assemble_page(&lines);
        assert!(text.is_empty());
        assert!(heading.is_none());
    }
}
