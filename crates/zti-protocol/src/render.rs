use std::fmt::Write as _;

use crate::response::{Confidence, SearchHit, SearchResults};

const DEFAULT_CHAR_BUDGET: usize = 80_000;

fn count_digits(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut d = 0;
    let mut v = n;
    while v > 0 {
        d += 1;
        v /= 10;
    }
    d
}

pub fn format_search_results(results: &SearchResults) -> String {
    format_search_results_budgeted(results, DEFAULT_CHAR_BUDGET)
}

pub fn format_search_results_budgeted(results: &SearchResults, budget: usize) -> String {
    if results.hits.is_empty() {
        return String::from("  no results\n");
    }

    let est = budget.min(
        256 + results
            .hits
            .iter()
            .chain(results.appendix.iter())
            .map(estimate_hit_bytes)
            .sum::<usize>(),
    );
    let mut out = String::with_capacity(est);

    match results.confidence {
        Confidence::High => {}
        Confidence::Medium => out.push_str(
            "⚠ medium confidence — try searchDep for exact symbols, or refine the query\n\n",
        ),
        Confidence::Low => out.push_str(
            "⚠ low confidence — try searchDep for exact symbols, or refine the query\n\n",
        ),
    }

    let _ = writeln!(
        out,
        "({} hits, {} related):\n",
        results.hits.len(),
        results.appendix.len()
    );

    let mut omitted = 0usize;

    for (i, hit) in results.hits.iter().enumerate() {
        let rank_len = 1 + count_digits(i + 1) + 1 + 6 + 1;
        let hdr_len = hit.file_path.len()
            + 1
            + count_digits(hit.start_line as usize)
            + 1
            + count_digits(hit.end_line as usize);
        let min_cost = rank_len + hdr_len + 1 + 64;
        if out.len() + min_cost > budget {
            omitted = results.hits.len() - i;
            break;
        }
        let _ = writeln!(out, "#{} {:.4}", i + 1, hit.score);
        write_hit_budgeted_inner(&mut out, hit, hdr_len, budget);
    }

    if !results.appendix.is_empty() && out.len() < budget {
        out.push_str("-- deps:\n");
        for hit in &results.appendix {
            let hdr_len = hit.file_path.len()
                + 1
                + count_digits(hit.start_line as usize)
                + 1
                + count_digits(hit.end_line as usize);
            let min_cost = hdr_len + 1 + 64;
            if out.len() + min_cost > budget {
                omitted += 1;
                continue;
            }
            write_hit_budgeted_inner(&mut out, hit, hdr_len, budget);
        }
    }

    if omitted > 0 {
        let notice =
            format!("\n... ({omitted} results omitted — reduce `limit` or narrow query)\n");
        if out.len() + notice.len() <= budget {
            out.push_str(&notice);
        }
    }

    out
}

fn estimate_hit_bytes(h: &SearchHit) -> usize {
    h.content.len() + h.file_path.len() + 64
}

fn write_hit_budgeted_inner(out: &mut String, hit: &SearchHit, hdr_len: usize, budget: usize) {
    if out.len() + hdr_len + 1 > budget {
        out.push_str("    [omitted]\n");
        return;
    }
    let _ = write!(out, "{}:{}-{}", hit.file_path, hit.start_line, hit.end_line);
    out.push('\n');
    let remaining = budget.saturating_sub(out.len());
    if remaining < 64 {
        out.push_str("    [omitted]\n");
        return;
    }
    let mut written = 0usize;
    for line in hit.content.lines() {
        let cost = 4 + line.len() + 1;
        if written + cost > remaining {
            out.push_str("    ...\n");
            return;
        }
        out.push_str("    ");
        out.push_str(line);
        out.push('\n');
        written += cost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_hit(name: &str, kind: &str, sym_id: u32, body: &str) -> SearchHit {
        SearchHit {
            chunk_id: [0u8; 16],
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
            total: 1,
            confidence: Confidence::High,
        };
        let out = format_search_results(&r);
        assert!(out.contains("(1 hits, 1 related)"), "header: {}", out);
        assert!(out.contains("#1 0.7407\n"), "hit rank line: {}", out);
        assert!(
            out.contains("src/poly/rq.rs:127-203\n"),
            "hit header: {}",
            out
        );
        assert!(out.contains("    pub fn recip"), "body indent: {}", out);
        assert!(out.contains("-- deps:\n"), "appendix marker: {}", out);
        assert!(
            out.contains("    pub fn i16_negative_mask"),
            "appendix body indent: {}",
            out
        );
    }

    #[test]
    fn empty_results_shows_no_results() {
        let r = SearchResults {
            hits: Vec::new(),
            appendix: Vec::new(),
            total: 0,
            confidence: Confidence::High,
        };
        let out = format_search_results(&r);
        assert!(out.contains("no results"), "{}", out);
    }

    #[test]
    fn budget_truncates_large_output() {
        let hits: Vec<SearchHit> = (0..50)
            .map(|i| mk_hit(&format!("fn_{i}"), "fn", i, &"x".repeat(500)))
            .collect();
        let r = SearchResults {
            hits,
            appendix: Vec::with_capacity(0),
            total: 50,
            confidence: Confidence::High,
        };
        let out = format_search_results_budgeted(&r, 500);
        assert!(out.len() <= 500, "budget overshoot: {}", out.len());
        assert!(out.contains("omitted"), "should note omitted results");
    }
}
