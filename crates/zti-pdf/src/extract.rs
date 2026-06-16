//! Per-page extraction orchestrator: load the document with `lopdf`, walk its
//! pages, interpret each page's content stream, and detect a heading. All
//! heavy lifting lives in [`crate::interpreter`] (text) and [`crate::heading`]
//! (analysis); this module is wiring only.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use lopdf::{Document, Encoding, ObjectId};

use crate::encoding::push_winansi;
use crate::heading::{body_font_size, pick_heading};
use crate::interpreter::{GlyphDecoder, interpret};

/// Per-font glyph decoder for one page. Maps each `/Resources /Font` resource
/// name to its lopdf [`Encoding`] (ToUnicode CMap, `/Differences`, or a base
/// encoding) and decodes shown text through it, recovering ligatures, math
/// symbols, and subscripts that the fixed WinAnsi byte map mangles.
struct PageDecoder<'a> {
    encodings: HashMap<Vec<u8>, Encoding<'a>>,
}

impl GlyphDecoder for PageDecoder<'_> {
    fn decode(&self, font: Option<&[u8]>, bytes: &[u8], out: &mut String) {
        let start = out.len();
        if let Some(name) = font
            && let Some(enc) = self.encodings.get(name)
            && enc.write_to_string(bytes, out).is_ok()
            && !is_mostly_replacement(&out[start..])
        {
            return;
        }
        // No usable encoding, decode error, or replacement-char garbage: drop
        // anything partially written and fall back to the Tier-1 byte map.
        out.truncate(start);
        push_winansi(out, bytes);
    }
}

/// True when more than a third of `s` is the Unicode replacement character —
/// the signature of a sparse/broken ToUnicode CMap whose codes mostly miss.
/// Such output is worse than the WinAnsi fallback, so we reject it.
fn is_mostly_replacement(s: &str) -> bool {
    let total = s.chars().count();
    if total == 0 {
        return false;
    }
    let bad = s.chars().filter(|&c| c == '\u{fffd}').count();
    bad * 3 > total
}

/// Build a page's font decoder from its resource dictionary. Fonts whose
/// encoding cannot be resolved are simply omitted (they fall back to WinAnsi).
fn page_decoder(doc: &Document, page_id: ObjectId) -> PageDecoder<'_> {
    let Ok(fonts) = doc.get_page_fonts(page_id) else {
        return PageDecoder {
            encodings: HashMap::new(),
        };
    };
    let mut encodings = HashMap::with_capacity(fonts.len());
    for (name, dict) in fonts {
        if let Ok(enc) = dict.get_font_encoding(doc) {
            encodings.insert(name, enc);
        }
    }
    PageDecoder { encodings }
}

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
        let decoder = page_decoder(&doc, page_id);
        let lines = interpret(&content, &decoder);
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
