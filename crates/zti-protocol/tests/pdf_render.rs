//! Specification tests for PDF search-result rendering.
//!
//! A `document`-kind hit renders as a titled, sectioned passage: the detected
//! heading becomes a `Title:` line and the body is split into `Description:`
//! and `Example:` blocks so a downstream LLM gets database-like fields rather
//! than a wall of prose. These tests pin that contract.

use zti_protocol::render::format_search_results;
use zti_protocol::response::{SearchHit, SearchResults};

fn pdf_hit(qualified: &str, body: &str) -> SearchHit {
    SearchHit {
        chunk_id: [0u8; 16],
        file_path: "taocp4f2.pdf".to_string(),
        symbol_qualified: qualified.to_string(),
        symbol_kind: "document".to_string(),
        sym_id: u32::MAX,
        start_line: 1,
        end_line: 1,
        content: body.to_string(),
        score: 0.8125,
    }
}

#[test]
fn pdf_hit_renders_header_with_page_range() {
    // The file:page-range header is preserved alongside the new structured
    // fields, and the body prose still survives in the Description block.
    let results = SearchResults {
        hits: vec![pdf_hit(
            "Algorithm C · p.1-1",
            "Permutation generation by cyclic shifts.\nExample: 1234, 2341.",
        )],
        appendix: vec![],
        total: 1,
    };
    let out = format_search_results(&results);
    assert!(out.contains("taocp4f2.pdf:1-1"), "header: {out}");
    assert!(out.contains("Permutation generation"), "body prose: {out}");
}

#[test]
fn pdf_hit_should_render_structured_title_section() {
    // The rendered output labels the heading as a structured "Title:" section
    // rather than only burying it in the file:line header.
    let results = SearchResults {
        hits: vec![pdf_hit(
            "Algorithm C · p.1-1",
            "Permutation generation by cyclic shifts.\nExample: 1234, 2341.",
        )],
        appendix: vec![],
        total: 1,
    };
    let out = format_search_results(&results);
    assert!(
        out.contains("Title:") && out.contains("Algorithm C"),
        "rendered PDF hit should surface a structured Title section: {out}"
    );
}

#[test]
fn pdf_hit_should_render_description_and_example_sections() {
    // When the chunk body contains an "Example:" lead, the renderer labels the
    // prose sections so a downstream LLM gets database-like fields, not a blob.
    let results = SearchResults {
        hits: vec![pdf_hit(
            "Algorithm C · p.1-1",
            "Permutation generation by cyclic shifts.\nExample: 1234, 2341.",
        )],
        appendix: vec![],
        total: 1,
    };
    let out = format_search_results(&results);
    assert!(out.contains("Description:"), "missing Description section: {out}");
    assert!(out.contains("Example:"), "missing Example section: {out}");
}
