use std::sync::LazyLock;

use regex::Regex;

pub(crate) struct SynLangConfig {
    pub separator_regex: Vec<Regex>,
}

pub(crate) static DEFAULT_LANG_CONFIG: LazyLock<SynLangConfig> = LazyLock::new(|| SynLangConfig {
    separator_regex: [
        r"\n\n+",
        r"\n",
        r"[\.\?!]\s+|。|？|！",
        r"[;:\-—]\s+|；|：|—+",
        r",\s+|，",
        r"\s+",
    ]
    .into_iter()
    .filter_map(|s| Regex::new(s).ok())
    .collect(),
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum LineBreakLevel {
    Inline,
    Newline,
    DoubleNewline,
}

impl LineBreakLevel {
    pub fn ord(&self) -> usize {
        match self {
            LineBreakLevel::Inline => 0,
            LineBreakLevel::Newline => 1,
            LineBreakLevel::DoubleNewline => 2,
        }
    }
}

pub(crate) fn line_break_level(text: &str) -> LineBreakLevel {
    let mut level = LineBreakLevel::Inline;
    let mut cs = text.chars();
    while let Some(c) = cs.next() {
        if c == '\n' || c == '\r' {
            level = LineBreakLevel::Newline;
            for c2 in cs.by_ref() {
                if c2 == '\n' || c2 == '\r' {
                    if c == c2 {
                        return LineBreakLevel::DoubleNewline;
                    }
                } else {
                    break;
                }
            }
        }
    }
    level
}

/// A single atomic chunk produced during Phase 1.
pub(crate) struct AtomChunk {
    pub byte_start: usize,
    pub byte_end: usize,
    pub boundary_syntax_level: usize,
    pub internal_lb_level: LineBreakLevel,
    pub boundary_lb_level: LineBreakLevel,
}

/// Collects atoms, trimming whitespace and aligning to line boundaries.
pub(crate) struct AtomCollector<'s> {
    pub text: &'s str,
    pub curr_level: usize,
    pub min_level: usize,
    pub chunks: Vec<AtomChunk>,
}

const SPACE: [char; 2] = [' ', '\t'];

impl<'s> AtomCollector<'s> {
    pub fn new(text: &'s str) -> Self {
        Self {
            text,
            curr_level: 0,
            min_level: 0,
            chunks: Vec::with_capacity(text.len() / 32 + 1),
        }
    }

    pub fn add(&mut self, start: usize, end: usize) {
        let trimmed = self.text[start..end].trim_end();
        if trimmed.is_empty() {
            return;
        }
        let trimmed_start = trimmed.trim_start();
        let ns = start + (trimmed.len() - trimmed_start.len());
        let ne = ns + trimmed_start.len();

        let prev = self.chunks.last().map_or(0, |c| c.byte_end);
        let gap = &self.text[prev..ns];
        let bl = line_break_level(gap);
        let (as_, ae) = if !matches!(bl, LineBreakLevel::Inline) {
            let g = gap.trim_end_matches(SPACE);
            (prev + g.len(), ne)
        } else {
            (ns, ne)
        };

        self.chunks.push(AtomChunk {
            byte_start: as_,
            byte_end: ae,
            boundary_syntax_level: self.min_level,
            internal_lb_level: line_break_level(trimmed_start),
            boundary_lb_level: bl,
        });
        self.min_level = self.curr_level;
    }

    pub fn seal(mut self) -> Vec<AtomChunk> {
        self.min_level = 0;
        self.chunks.push(AtomChunk {
            byte_start: self.text.len(),
            byte_end: self.text.len(),
            boundary_syntax_level: self.min_level,
            internal_lb_level: LineBreakLevel::Inline,
            boundary_lb_level: LineBreakLevel::DoubleNewline,
        });
        self.chunks
    }
}
