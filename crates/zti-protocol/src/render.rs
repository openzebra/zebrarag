use std::fmt::Write as _;

use crate::response::{SearchHit, SearchResults};

/// Format `SearchResults` for chat output. Compact layout — body already
/// contains the function signature, so the header is just `kind#id path:lines`
/// with no `sig/→/>/≈` lines. Single allocation up front sized from the
/// payload — no per-line `println!`.
///
/// ```text
/// LEGEND ...
/// #1 0.7407 recip
/// m#183 src/poly/rq.rs:127-203
///     pub fn recip(...) -> ... { ... }
/// --- APPENDIX ---
/// f#97 src/math/nums.rs:21-26
///     pub fn i16_negative_mask(...) -> i16 { ... }
/// ```
pub fn format_search_results(results: &SearchResults) -> String {
    let est = 256
        + results.legend.len()
        + results
            .hits
            .iter()
            .chain(results.appendix.iter())
            .map(estimate_hit_bytes)
            .sum::<usize>();
    let mut out = String::with_capacity(est);

    out.push_str(&results.legend);
    out.push('\n');

    if results.hits.is_empty() {
        out.push_str("  no results\n");
        return out;
    }

    for (i, hit) in results.hits.iter().enumerate() {
        let _ = writeln!(out, "#{} {:.4} {}", i + 1, hit.score, short_name(&hit.symbol_qualified));
        write_hit_block(&mut out, hit);
    }

    if !results.appendix.is_empty() {
        out.push_str("--- APPENDIX ---\n");
        for hit in &results.appendix {
            write_hit_block(&mut out, hit);
        }
    }

    out
}

fn estimate_hit_bytes(h: &SearchHit) -> usize {
    h.content.len() + h.file_path.len() + 64
}

fn write_hit_block(out: &mut String, hit: &SearchHit) {
    let _ = writeln!(
        out,
        "{}#{} {}:{}-{}",
        kind_short(&hit.symbol_kind),
        hit.sym_id,
        hit.file_path,
        hit.start_line,
        hit.end_line
    );
    for line in hit.content.lines() {
        out.push_str("    ");
        out.push_str(line);
        out.push('\n');
    }
}

/// Last `::` segment of a qualified name, or the whole string if there
/// is no separator. Borrows from `q` — no allocation.
fn short_name(q: &str) -> &str {
    match q.rsplit_once("::") {
        Some((_, tail)) => tail,
        None => q,
    }
}

/// Map long-form symbol kind ("fn", "method", …) to the short prefix
/// used in DSL output ("f", "m", …). Matches `zti_ts_core::types::Kind::short`
/// without taking the dependency.
fn kind_short(kind: &str) -> &'static str {
    match kind {
        "fn" => "f",
        "method" => "m",
        "struct" => "s",
        "enum" => "e",
        "class" => "C",
        "interface" => "I",
        "typealias" => "t",
        "const" => "c",
        "static" => "v",
        "module" => "M",
        "field" | "variant" => ".",
        "event" => "E",
        "error" => "X",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    fn mk_hit(name: &str, kind: &str, sym_id: u32, body: &str) -> SearchHit {
        SearchHit {
            chunk_id: Vec::new(),
            file_path: "src/poly/rq.rs".to_string(),
            symbol_qualified: format!("foo::{}", name),
            symbol_kind: kind.to_string(),
            sym_id,
            start_line: 127,
            end_line: 203,
            content: body.to_string(),
            score: 0.7407,
        }
    }

    #[test]
    fn format_compact_shape() {
        let r = SearchResults {
            hits: vec![mk_hit(
                "recip",
                "method",
                183,
                "pub fn recip(...) -> Result {\n    body\n}",
            )],
            appendix: vec![mk_hit(
                "i16_negative_mask",
                "fn",
                97,
                "pub fn i16_negative_mask(x: i16) -> i16 { -(x as i16) }",
            )],
            legend: Cow::Borrowed("LEGEND test"),
            total: 1,
        };
        let out = format_search_results(&r);
        assert!(out.starts_with("LEGEND test\n"), "legend prefix: {}", out);
        assert!(out.contains("#1 0.7407 recip\n"), "hit rank line: {}", out);
        assert!(out.contains("m#183 src/poly/rq.rs:127-203\n"), "hit header: {}", out);
        assert!(out.contains("    pub fn recip"), "body indent: {}", out);
        assert!(out.contains("--- APPENDIX ---\n"), "appendix marker: {}", out);
        assert!(out.contains("f#97 src/poly/rq.rs:127-203\n"), "appendix header: {}", out);
        assert!(out.contains("    pub fn i16_negative_mask"), "appendix body indent: {}", out);
        // No more sig/→/>/≈ lines.
        assert!(!out.contains("sig "), "should have no sig line: {}", out);
        assert!(!out.contains("  ---\n"), "should have no separator: {}", out);
    }

    #[test]
    fn empty_results_shows_no_results() {
        let r = SearchResults {
            hits: Vec::new(),
            appendix: Vec::new(),
            legend: Cow::Borrowed("LEGEND"),
            total: 0,
        };
        let out = format_search_results(&r);
        assert!(out.contains("no results"), "{}", out);
    }

    #[test]
    fn short_name_takes_last_segment() {
        assert_eq!(short_name("foo::bar::baz"), "baz");
        assert_eq!(short_name("baz"), "baz");
        assert_eq!(short_name(""), "");
    }

    #[test]
    fn kind_short_maps_known_kinds() {
        assert_eq!(kind_short("fn"), "f");
        assert_eq!(kind_short("method"), "m");
        assert_eq!(kind_short("struct"), "s");
        assert_eq!(kind_short("unknown"), "?");
    }
}
