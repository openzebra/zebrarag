pub mod chunk_strategy;
pub mod dsl;
pub mod file_type;
pub mod format;
pub mod ids;
pub mod paths;

use std::ops::Range;

/// Map an inclusive 1-based line range `[start_line, end_line]` to a byte
/// range over `src`. Lines are determined by `'\n'` positions; the returned
/// range never exceeds `src.len()` and is empty when the requested range is
/// out of bounds.
///
/// Cheaper than `src.lines().collect::<Vec<_>>()[..].join("\n")` — single
/// pass over the source, one slice, no allocations.
pub fn line_byte_range(src: &str, start_line_1based: u32, end_line_1based: u32) -> Range<usize> {
    if start_line_1based == 0 || end_line_1based < start_line_1based {
        return 0..0;
    }
    let start_target = (start_line_1based - 1) as usize;
    let end_target = end_line_1based as usize;

    let mut line = 0usize;
    let mut start_byte: Option<usize> = None;
    let mut end_byte = src.len();

    if start_target == 0 {
        start_byte = Some(0);
    }

    for (i, _) in src.match_indices('\n') {
        line += 1;
        if start_byte.is_none() && line == start_target {
            start_byte = Some(i + 1);
        }
        if line == end_target {
            end_byte = i;
            break;
        }
    }

    let start = start_byte.unwrap_or(src.len());
    if start > end_byte {
        return start..start;
    }
    start..end_byte
}

#[derive(Debug, Clone, Default)]
pub struct LineIndex {
    line_starts: Vec<usize>,
    len: usize,
}

impl LineIndex {
    pub fn new(src: &str) -> Self {
        let mut line_starts = Vec::with_capacity(src.len() / 24 + 1);
        line_starts.push(0);
        for (i, _) in src.match_indices('\n') {
            line_starts.push(i + 1);
        }
        Self {
            line_starts,
            len: src.len(),
        }
    }

    pub fn byte_range(&self, start_line_1based: u32, end_line_1based: u32) -> Range<usize> {
        if start_line_1based == 0 || end_line_1based < start_line_1based {
            return 0..0;
        }
        let start = self
            .line_starts
            .get((start_line_1based - 1) as usize)
            .copied()
            .unwrap_or(self.len);
        let end = match self.line_starts.get(end_line_1based as usize) {
            Some(&nl) => nl.saturating_sub(1),
            None => self.len,
        };
        if start > end {
            return start..start;
        }
        start..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_byte_range_basic() {
        let s = "alpha\nbeta\ngamma\ndelta";
        // line 1 only ("alpha")
        let r = line_byte_range(s, 1, 1);
        assert_eq!(&s[r], "alpha");
        // lines 2..=3 ("beta\ngamma")
        let r = line_byte_range(s, 2, 3);
        assert_eq!(&s[r], "beta\ngamma");
        // last line, no trailing newline
        let r = line_byte_range(s, 4, 4);
        assert_eq!(&s[r], "delta");
    }

    #[test]
    fn line_byte_range_out_of_bounds() {
        let s = "one\ntwo";
        let r = line_byte_range(s, 0, 1);
        assert_eq!(r, 0..0);
        let r = line_byte_range(s, 5, 7);
        assert!(r.is_empty());
    }

    #[test]
    fn line_index_matches_line_byte_range() {
        let cases: &[&str] = &[
            "",
            "hello",
            "alpha\nbeta\ngamma\ndelta",
            "one\n",
            "\n\n\n",
            "a\nb\nc\nd\ne\nf",
            "trailing\nnewline\n",
        ];

        for &src in cases {
            let idx = LineIndex::new(src);
            for start in 0u32..8 {
                for end in 0u32..8 {
                    let expected = line_byte_range(src, start, end);
                    let got = idx.byte_range(start, end);
                    assert_eq!(got, expected, "src={:?} start={} end={}", src, start, end);
                }
            }
        }
    }
}
