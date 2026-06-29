//! Integration test for the PDF packing pipeline: `extract_pages`-shaped
//! `PageText` structs → `assemble_pdf_contents` → `pack_pdf_pages` → chunks.
//!
//! This exercises the real packing path the indexer uses. The `PageText`
//! inputs mirror what `zrag_pdf::extract_pages` emits for the synthetic fixture
//! (validated in `zrag-pdf/tests/extract.rs`), so this is effectively a
//! round-trip without re-parsing the PDF bytes here.
//!
//! Run with: `cargo test -p zrag-pipeline --features testing`

#![cfg(feature = "testing")]

use zrag_pdf::PageText;
use zrag_pipeline::pdf_chunk::pack_pdf_pages;
use zrag_pipeline::testing::assemble_pdf_contents;

/// Mirror of the synthetic fixture's extracted output: a clean heading page,
/// a glue-bug page, and a noise-heading page.
fn fixture_pages() -> Vec<PageText> {
    vec![
        PageText {
            page: 1,
            text: "Algorithm C\nPermutation generation by cyclic shifts.\nExample: 1234, 2341, 3412.".into(),
            heading: Some("Algorithm C".into()),
        },
        PageText {
            page: 2,
            // The extractor currently glues these; packing must still carry
            // whatever text it is given.
            text: "Permutationgeneration".into(),
            heading: None,
        },
        PageText {
            page: 3,
            text: "x70.[M33]\nSome real body text here.".into(),
            // The extractor currently mis-detects this as a heading; packing
            // consumes it as-is (Phase 2 will fix detection upstream).
            heading: Some("x70.[M33]".into()),
        },
    ]
}

#[test]
fn assemble_joins_pages_with_form_feed_separators() {
    let pages = fixture_pages();
    let (contents, metas) = assemble_pdf_contents(&pages);
    assert_eq!(metas.len(), 3, "one meta per page");
    assert!(
        contents.contains('\u{c}'),
        "page boundaries must be marked by form-feed for pack_pdf_pages: {contents:?}"
    );
    // Headings flow through into meta parallel to the segments.
    assert_eq!(metas[0].heading.as_deref(), Some("Algorithm C"));
    assert!(metas[1].heading.is_none());
}

#[test]
fn pack_produces_one_chunk_when_under_budget() {
    let pages = fixture_pages();
    let (contents, metas) = assemble_pdf_contents(&pages);
    let chunks = pack_pdf_pages("doc.pdf", "/abs/doc.pdf", &contents, &metas, 10_000);
    assert_eq!(chunks.len(), 1, "small fixture packs into one chunk");
    let c = &chunks[0];
    // start_line/end_line carry page numbers (1..=3), not source lines.
    assert_eq!(c.start_line, 1);
    assert_eq!(c.end_line, 3);
    // The opening heading (page 1's) becomes the qualified label.
    assert!(
        c.qualified.contains("Algorithm C"),
        "qualified should carry the opening heading: {}",
        c.qualified
    );
    assert!(c.qualified.contains("p.1-3"), "qualified: {}", c.qualified);
    // All three pages' text survives into the body, joined by blank lines.
    assert!(c.body.contains("Permutation generation by cyclic shifts."));
    assert!(c.body.contains("Some real body text here."));
    // Form-feed separators must NOT leak into chunk bodies.
    assert!(!c.body.contains('\u{c}'));
}

#[test]
fn pack_splits_at_heading_once_past_half_budget() {
    // Force a small budget so page 1 alone crosses half-budget; page 3 opens
    // with a (mis)detected heading, so it must start a fresh chunk.
    let pages = fixture_pages();
    let (contents, metas) = assemble_pdf_contents(&pages);
    let chunks = pack_pdf_pages("doc.pdf", "/abs/doc.pdf", &contents, &metas, 120);
    assert!(
        chunks.len() >= 2,
        "expected a heading-aware split, got {} chunks",
        chunks.len()
    );
    // The last chunk should be page 3's heading-titled segment.
    let last = chunks.last().unwrap();
    assert_eq!(last.start_line, 3);
}

#[test]
fn pack_carries_page_numbers_through_empty_pages() {
    // Page 2 is blank: dropped from body, but page numbering must still reach 3.
    let pages = vec![
        PageText { page: 1, text: "real".into(), heading: None },
        PageText { page: 2, text: "   ".into(), heading: None },
        PageText { page: 3, text: "more".into(), heading: None },
    ];
    let (contents, metas) = assemble_pdf_contents(&pages);
    let chunks = pack_pdf_pages("d.pdf", "/abs/d.pdf", &contents, &metas, 10_000);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_line, 1);
    assert_eq!(chunks[0].end_line, 3);
}

// NB: the heading is surfaced as a structured "Title:" field at *render* time
// (see `zrag-protocol/tests/pdf_render.rs`), not prepended into the chunk body —
// so the stored/embedded text is unchanged and PDFs need no reindex.
