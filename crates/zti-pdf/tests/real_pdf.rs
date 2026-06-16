//! Real-world characterization against the Knuth TAOCP vol.4 fascicle 2 PDF.
//!
//! These tests are `#[ignore]`'d and gated on `ZTI_PDF_FIXTURE_DIR` so they
//! never run in CI (the 905 KB PDF is not redistributable and not committed).
//! Run locally with:
//!
//! ```sh
//! ZTI_PDF_FIXTURE_DIR=/Users/hicaru/projects/zebra/test \
//!   cargo test -p zti-pdf -- --ignored real_pdf
//! ```
//!
//! They exist to (a) freeze the current garbled extraction as a baseline and
//! (b) give Phase 2 a concrete readability target to diff against.

use std::path::PathBuf;
use zti_pdf::extract_pages;

/// Resolve the first `*.pdf` under `ZTI_PDF_FIXTURE_DIR`, or skip the test.
fn fixture_pdf() -> PathBuf {
    let dir = std::env::var("ZTI_PDF_FIXTURE_DIR")
        .expect("set ZTI_PDF_FIXTURE_DIR to the dir holding the Knuth PDF");
    let entry = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {dir}: {e}"))
        .flatten()
        .find(|e| {
            e.path().extension().and_then(|x| x.to_str()).is_some_and(|x| x.eq_ignore_ascii_case("pdf"))
        })
        .unwrap_or_else(|| panic!("no .pdf under {dir}"));
    entry.path()
}

/// Fraction of whitespace (space) chars in `s`. Readable English prose runs
/// ~0.15–0.20; the current extractor's glued output (no inter-word spaces from
/// `Td` X-gaps) scores well below 0.05. This is the metric that captures the
/// Knuth noise — `letter_ratio` is useless here because glued text is still
/// mostly alphanumeric.
fn space_ratio(s: &str) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    let spaces = s.chars().filter(|c| *c == ' ').count();
    spaces as f32 / s.len() as f32
}

#[test]
#[ignore = "requires ZTI_PDF_FIXTURE_DIR pointing at the Knuth PDF; not committed"]
fn real_pdf_extracts_all_pages_non_empty() {
    // CHARACTERIZATION: the extractor produces one entry per page and the bulk
    // of pages yield *some* text (even if glued). This is the floor.
    let bytes = std::fs::read(fixture_pdf()).unwrap();
    let pages = extract_pages(&bytes).expect("Knuth PDF must load");
    assert!(pages.len() > 50, "TAOCP 4f2 has many pages, got {}", pages.len());
    let non_empty = pages.iter().filter(|p| !p.text.trim().is_empty()).count();
    assert!(
        non_empty > pages.len() / 2,
        "most pages should yield text: {non_empty}/{} non-empty",
        pages.len()
    );
}

#[test]
#[ignore = "requires ZTI_PDF_FIXTURE_DIR; characterizes the fixed inter-word spacing"]
fn real_pdf_inter_word_spacing_is_above_floor() {
    // CHARACTERIZATION (passes after the inter-word-spacing fix): every
    // substantial page now has a space ratio >= 0.08, meaning the interpreter
    // correctly inserts spaces for horizontal `Td` repositioning. The previous
    // behaviour glued words together (space ratio ~0.000).
    let bytes = std::fs::read(fixture_pdf()).unwrap();
    let pages = extract_pages(&bytes).unwrap();
    for (i, page) in pages.iter().enumerate() {
        if page.text.len() < 100 {
            continue;
        }
        let ratio = space_ratio(&page.text);
        assert!(
            ratio >= 0.08,
            "page {} space ratio {ratio:.3} < 0.08 — regression in inter-word spacing",
            i + 1
        );
    }
}
