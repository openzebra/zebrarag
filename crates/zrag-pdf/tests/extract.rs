//! Integration tests for `zrag_pdf::extract_pages` against a hand-authored
//! synthetic PDF fixture.
//!
//! Tests are split into two groups:
//! - **characterization** (`*_current_*`): freeze the behaviour the code has
//!   *today*, including bugs, so a Phase-2 fix shows up as a red test that
//!   tells you exactly which assumption moved.
//! - **specification** (`*_should_*`): assert the behaviour we *want*. These
//!   fail today and document the Phase-2 work items. They are `#[ignore]`'d so
//!   `cargo test` stays green; run them with `cargo test -- --ignored`.

mod common;

use zrag_pdf::extract_pages;

fn fixture() -> Vec<u8> {
    common::build_sample_pdf()
}

#[test]
fn fixture_loads_and_has_three_pages() {
    let pages = extract_pages(&fixture()).expect("fixture must be a loadable PDF");
    assert_eq!(pages.len(), 3, "fixture has 3 pages");
}

// ---- Page 1: baseline (good page) ---------------------------------------

#[test]
fn page1_heading_is_algorithm_c() {
    let pages = extract_pages(&fixture()).unwrap();
    let p1 = &pages[0];
    assert_eq!(
        p1.heading.as_deref(),
        Some("Algorithm C"),
        "the 18pt oversized line must be detected as the page heading"
    );
}

#[test]
fn page1_body_contains_prose_and_example() {
    let pages = extract_pages(&fixture()).unwrap();
    let text = &pages[0].text;
    assert!(text.contains("Permutation generation by cyclic shifts."), "got: {text}");
    assert!(text.contains("Example: 1234, 2341, 3412."), "got: {text}");
}

// ---- Page 2: inter-word spacing (the "Permutationgeneration" bug) --------
//
// PDF places words with horizontal `Tj`+`Td` gaps and often no space byte.
// The interpreter now inserts a space when the X position advances beyond a
// quarter of the font size since the last glyph.

#[test]
fn page2_inserts_space_between_tj_words() {
    let pages = extract_pages(&fixture()).unwrap();
    assert!(
        pages[1].text.contains("Permutation generation"),
        "a horizontal Td gap between Tj ops must produce a space: {:?}",
        pages[1].text
    );
}

#[test]
fn page2_does_not_double_space() {
    // Regression guard: the inter-word fix must not introduce leading or
    // trailing spaces, or double spaces within the text.
    let pages = extract_pages(&fixture()).unwrap();
    let text = &pages[1].text;
    assert!(!text.starts_with(' '), "no leading space: {text:?}");
    assert!(!text.ends_with(' '), "no trailing space: {text:?}");
    assert!(!text.contains("  "), "no double spaces: {text:?}");
}

// ---- Page 3: heading validation (the "x70.[M33]" bug) -------------------
//
// `pick_heading` returns the first oversized short line. A digit/punctuation
// heavy line at heading size is noise, not a title.

#[test]
fn page3_rejects_digit_heavy_heading() {
    // The letter-ratio guard (≥0.5 alphabetic) drops the 18pt `x70.[M33]` line:
    // 2 letters / 9 chars = 0.22. No other line is heading-sized, so the page
    // has no heading.
    let pages = extract_pages(&fixture()).unwrap();
    let heading = pages[2].heading.as_deref();
    assert!(
        heading.is_none_or(|h| !h.contains("x70")),
        "a digit/punct-heavy oversized line must not become a heading, got: {heading:?}"
    );
}
