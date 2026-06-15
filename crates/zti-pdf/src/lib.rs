//! Tier-1 PDF text extraction with page-level metadata.
//!
//! Decodes each page's content stream (BT/ET text blocks, `Tf` font size,
//! `Tj`/`TJ`/`'`/`"` text show, `Td`/`TD`/`T*`/`Tm` positioning) and emits
//! one [`PageText`] per page with the plain text plus a heading detected via
//! font-size clustering. Latin PDFs (WinAnsiEncoding / Standard-14 fonts) are
//! the supported tier; CID/ToUnicode fonts and encrypted documents are Tier-2+.
//!
//! # Module layout
//!
//! - [`encoding`] — WinAnsiEncoding byte→char map.
//! - [`tokenizer`] — low-level content-stream operand primitives.
//! - [`interpreter`] — text state machine that emits [`interpreter::Line`]s.
//! - [`heading`] — font-size clustering to pick a page heading.
//! - [`extract`] — orchestrator: `lopdf` load + per-page wiring.

mod encoding;
mod extract;
mod heading;
mod interpreter;
mod tokenizer;

pub use extract::PageText;

use anyhow::Result;

/// Extract per-page text and detected headings from a PDF byte stream.
///
/// Returns one entry per page in ascending page order. Pages whose content
/// stream cannot be parsed yield an empty `text` and `None` heading rather
/// than aborting the whole document; callers should warn-and-skip pages or
/// the whole file as appropriate when every page comes back empty.
pub fn extract_pages(bytes: &[u8]) -> Result<Vec<PageText>> {
    extract::extract_pages(bytes)
}
