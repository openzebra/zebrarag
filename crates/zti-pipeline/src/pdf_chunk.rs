//! Page-aware PDF chunking.
//!
//! Packs a PDF's form-feed-separated page text into book-like [`Chunk`]s.
//! Pages are greedily accumulated until either the byte `budget` is reached
//! (hard boundary) or a page that opens a heading arrives while the current
//! chunk is already past half-budget (heading-aware boundary). Each chunk's
//! `start_line`/`end_line` carry the first/last page number it spans, and
//! `qualified` carries the opening heading (`"Chapter 3 · p.42-58"`) or a bare
//! page range (`"p.42-58"`) when no heading opened the chunk.

use std::borrow::Cow;
use std::sync::Arc;

use zti_dsl::chunking::{Chunk, ChunkStrategy};
use zti_ts_core::types::Kind;

use crate::manifest::PdfPageMeta;

/// Pack form-feed-separated page text into book-like chunks.
///
/// `contents` uses `\n\u{c}\n` (form-feed) page separators; `page_metas` holds
/// one [`PdfPageMeta`] per segment, parallel to the page index. The `budget`
/// is a byte packing hint — the indexer's `adaptive_split` still re-splits any
/// chunk that exceeds the embed model's token window downstream.
pub fn pack_pdf_pages<'a>(
    rel: &str,
    full_path: &str,
    contents: &'a str,
    page_metas: &[PdfPageMeta],
    budget: usize,
) -> Vec<Chunk<'a>> {
    let half_budget = budget / 2;
    let mut out: Vec<Chunk<'a>> = Vec::new();
    let mut buf = String::new();
    let mut start_page: Option<u32> = None;
    let mut end_page: u32 = 0;
    let mut heading: Option<String> = None;

    let flush = |buf: &mut String,
                 start_page: Option<u32>,
                 end_page: u32,
                 heading: &Option<String>,
                 out: &mut Vec<Chunk<'a>>| {
        if buf.is_empty() {
            return;
        }
        let chunk = build_pdf_chunk(
            rel,
            full_path,
            std::mem::take(buf),
            start_page.unwrap_or(1),
            end_page,
            heading.as_deref(),
        );
        out.push(chunk);
    };

    for (i, raw) in contents.split('\u{c}').enumerate() {
        let page_num = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let page_heading = page_metas.get(i).and_then(|m| m.heading.as_deref());
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Heading-aware boundary: start a fresh chunk at a heading once the
        // current chunk is meaningfully full.
        if page_heading.is_some() && start_page.is_some() && buf.len() >= half_budget {
            flush(&mut buf, start_page, end_page, &heading, &mut out);
            start_page = None;
            heading = None;
        }

        if start_page.is_none() {
            start_page = Some(page_num);
            heading = page_heading.map(str::to_string);
        }
        if !buf.is_empty() {
            buf.push_str("\n\n");
        }
        buf.push_str(trimmed);
        end_page = page_num;

        // Hard budget boundary.
        if buf.len() >= budget {
            flush(&mut buf, start_page, end_page, &heading, &mut out);
            start_page = None;
            heading = None;
        }
    }

    flush(&mut buf, start_page, end_page, &heading, &mut out);
    out
}

/// Assemble one PDF chunk. `start_page`/`end_page` overload the chunk's
/// `start_line`/`end_line` (PDFs have no source lines); `heading` becomes the
/// human-readable `qualified` label surfaced in search results.
fn build_pdf_chunk(
    rel: &str,
    full_path: &str,
    body: String,
    start_page: u32,
    end_page: u32,
    heading: Option<&str>,
) -> Chunk<'static> {
    let qualified = match heading {
        Some(h) => format!("{h} · p.{start_page}-{end_page}"),
        None => format!("p.{start_page}-{end_page}"),
    };
    Chunk {
        file: Arc::from(full_path),
        rel_file: Arc::from(rel),
        start_line: start_page,
        end_line: end_page,
        sym_id: u32::MAX,
        sub_chunk_idx: 0,
        total_sub_chunks: 1,
        chunk_strategy: ChunkStrategy::Symbol,
        body: Cow::Owned(body),
        qualified,
        kind: Kind::Document,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::PdfPageMeta;

    #[test]
    fn small_pdf_is_one_chunk_spanning_all_pages() {
        let contents = "page one text\n\u{c}\npage two text";
        let metas = vec![
            PdfPageMeta {
                heading: Some("Intro".into()),
            },
            PdfPageMeta { heading: None },
        ];
        let chunks = pack_pdf_pages("doc.pdf", "/abs/doc.pdf", contents, &metas, 10_000);
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        // Page numbers overload start_line/end_line.
        assert_eq!(c.start_line, 1);
        assert_eq!(c.end_line, 2);
        assert_eq!(c.qualified, "Intro · p.1-2");
        assert_eq!(c.kind, Kind::Document);
        assert!(c.body.contains("page one text"));
        assert!(c.body.contains("page two text"));
        // Form-feed separators are NOT in the chunk body — pages join on \n\n.
        assert!(!c.body.contains('\u{c}'));
    }

    #[test]
    fn splits_at_heading_past_half_budget() {
        // budget 100 → half 50. Page 1 alone (60 chars) crosses half; page 2
        // opens a heading, so it must start a fresh chunk.
        let big = "x".repeat(60);
        let contents = format!("{big}\n\u{c}\npage two");
        let metas = vec![
            PdfPageMeta { heading: None },
            PdfPageMeta {
                heading: Some("Chapter 2".into()),
            },
        ];
        let chunks = pack_pdf_pages("doc.pdf", "/abs/doc.pdf", &contents, &metas, 100);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 1);
        assert_eq!(chunks[0].qualified, "p.1-1");
        assert_eq!(chunks[1].start_line, 2);
        assert_eq!(chunks[1].end_line, 2);
        assert_eq!(chunks[1].qualified, "Chapter 2 · p.2-2");
    }

    #[test]
    fn flushes_at_hard_budget() {
        // budget 20; page 1 is 25 chars → hard boundary fires after page 1.
        let p1 = "a".repeat(25);
        let contents = format!("{p1}\n\u{c}\nshort");
        let metas = vec![PdfPageMeta { heading: None }, PdfPageMeta { heading: None }];
        let chunks = pack_pdf_pages("d.pdf", "/abs/d.pdf", &contents, &metas, 20);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 1);
        assert_eq!(chunks[1].start_line, 2);
    }

    #[test]
    fn skips_empty_pages_but_keeps_page_numbers() {
        // Page 2 is whitespace-only; it is dropped from the body but page
        // numbering still reflects the real page index (1 and 3).
        let contents = "real\n\u{c}\n   \n\u{c}\nmore";
        let metas = vec![
            PdfPageMeta { heading: None },
            PdfPageMeta { heading: None },
            PdfPageMeta { heading: None },
        ];
        let chunks = pack_pdf_pages("d.pdf", "/abs/d.pdf", contents, &metas, 10_000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
        assert!(chunks[0].body.contains("real"));
        assert!(chunks[0].body.contains("more"));
    }

    #[test]
    fn no_heading_yields_bare_page_range() {
        let contents = "only page";
        let metas = vec![PdfPageMeta { heading: None }];
        let chunks = pack_pdf_pages("d.pdf", "/abs/d.pdf", contents, &metas, 10_000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].qualified, "p.1-1");
    }
}
