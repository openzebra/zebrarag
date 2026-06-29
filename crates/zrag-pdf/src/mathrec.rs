//! Geometric reconstruction of matrices and vectors from positioned glyph runs.
//!
//! TAOCP-style PDFs render matrices with Computer-Modern math-extension Type3
//! fonts whose glyph names are obfuscated and whose ToUnicode CMap is an
//! identity range. The flat text path therefore yields gibberish: extensible
//! delimiters decode to `0BBBBB@` / `1CCCCCA`, rows glue into one line, and
//! minus signs (byte `0x00`) vanish. The *geometry* survives intact, so this
//! module rebuilds the grid from [`PosRun`] coordinates:
//!
//! - A left paren is a same-x glyph stack spelling `0` `B`* `@` (top→bottom); a
//!   right paren spells `1` `C`* `A`. The single-row forms are `0@` / `1A`.
//! - Body glyphs between a matched pair cluster into rows by Y and columns by
//!   the per-row X order. A glyph that decoded to nothing sitting left of a digit
//!   is the lost minus sign.
//!
//! Output per matrix is a dual block — Unicode box-art plus a `$$\begin{pmatrix}
//! …$$` LaTeX expression — spliced into the page text by [`crate::interpreter::assemble`]
//! so it is both displayed and embedded. Anything that fails validation is left
//! untouched as ordinary prose (lossless). Bracket/brace delimiters, summations
//! and systems are out of scope (no calibration data; the fonts hide their
//! symbols), but `Delim` leaves room to add them.

use std::borrow::Cow;
use std::fmt::{self, Write as _};

use crate::interpreter::{Line, PosRun, Region, RunPage, assemble};

/// Same-x tolerance (PDF units) for grouping a delimiter glyph stack.
const X_TOL: f32 = 2.0;
/// Minimum Y gap that can separate two matrix rows (below this is intra-row jitter).
const ROW_MIN_GAP: f32 = 3.0;

/// Which fence surrounds the array. Only `Paren` is calibrated today; the others
/// are reserved so bracket/brace forms can be added without reshaping callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delim {
    Paren,
}

impl Delim {
    /// The `\begin{…}` / `\end{…}` environment name for this fence.
    const fn env(self) -> &'static str {
        match self {
            Self::Paren => "pmatrix",
        }
    }
}

/// One detected delimiter glyph stack and the contiguous runs it spans.
#[derive(Debug)]
struct Fence {
    side: Side,
    delim: Delim,
    run_lo: usize,
    run_hi: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// A reconstructed array: rows of cells, each cell borrowed from the arena
/// unless a recovered sign forced an owned `-n` string.
#[derive(Debug)]
struct Matrix<'a> {
    delim: Delim,
    rows: Vec<Vec<Cow<'a, str>>>,
}

/// Detect every reconstructable matrix/vector and render each page with the
/// originals replaced by their dual box-art + LaTeX block.
#[must_use]
pub fn rewrite(page: &RunPage) -> Vec<Line> {
    let regions = detect(&page.runs, &page.arena);
    assemble(&page.runs, &page.arena, regions)
}

/// The decoded glyphs of one run (empty when the glyph decoded to nothing).
fn run_text<'a>(arena: &'a str, r: &PosRun) -> &'a str {
    arena.get(r.t0 as usize..r.t1 as usize).unwrap_or_default()
}

/// Find all paren-delimited matrices and turn each into a spliceable [`Region`],
/// sorted by start run and non-overlapping.
fn detect(runs: &[PosRun], arena: &str) -> Vec<Region> {
    let fences = fences(runs, arena);
    let mut regions: Vec<Region> = Vec::with_capacity(fences.len() / 2);
    let mut pending: Option<&Fence> = None;
    for fence in &fences {
        match fence.side {
            Side::Left => pending = Some(fence),
            Side::Right => {
                if let Some(left) = pending.take()
                    && left.delim == fence.delim
                    && fence.run_lo > left.run_hi
                    && let Some(body) = runs.get(left.run_hi + 1..fence.run_lo)
                    && let Some(matrix) = Matrix::from_body(body, arena, left.delim)
                {
                    let size = body.first().map_or(0.0, |r| r.size);
                    let mut block = matrix.to_string();
                    // Box-art first, then the LaTeX expression on its own lines.
                    let _ = write!(block, "\n{}", Latex(&matrix));
                    regions.push(Region {
                        run_lo: left.run_lo,
                        run_hi: fence.run_hi,
                        size,
                        block,
                    });
                }
            }
        }
    }
    regions
}

/// Scan runs for delimiter glyph stacks: maximal groups of consecutive runs that
/// share an X column and spell a paren fence top-to-bottom.
fn fences(runs: &[PosRun], arena: &str) -> Vec<Fence> {
    let mut out: Vec<Fence> = Vec::new();
    let mut i = 0usize;
    while i < runs.len() {
        let Some(first) = runs.get(i) else { break };
        let x = first.x;
        let mut j = i + 1;
        while runs.get(j).is_some_and(|r| (r.x - x).abs() <= X_TOL) {
            j += 1;
        }
        if let Some(cluster) = runs.get(i..j)
            && let Some((side, delim)) = classify(cluster, arena)
        {
            out.push(Fence {
                side,
                delim,
                run_lo: i,
                run_hi: j - 1,
            });
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Classify a same-x glyph cluster as a left/right paren fence, or `None`.
fn classify(cluster: &[PosRun], arena: &str) -> Option<(Side, Delim)> {
    let mut items: Vec<(f32, &str)> = cluster
        .iter()
        .map(|r| (r.y, run_text(arena, r)))
        .collect();
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    let chars: Vec<char> = items.iter().flat_map(|(_, t)| t.chars()).collect();
    let (&first, &last) = (chars.first()?, chars.last()?);
    if chars.len() < 2 {
        return None;
    }
    let middle = chars.get(1..chars.len() - 1).unwrap_or_default();
    match (first, last) {
        ('0', '@') if middle.iter().all(|c| *c == 'B') => Some((Side::Left, Delim::Paren)),
        ('1', 'A') if middle.iter().all(|c| *c == 'C') => Some((Side::Right, Delim::Paren)),
        _ => None,
    }
}

impl<'a> Matrix<'a> {
    /// Rebuild a matrix from the body runs between a matched fence pair, or
    /// `None` if the grid is not uniform (e.g. multi-token cells we don't model).
    fn from_body(body: &'a [PosRun], arena: &'a str, delim: Delim) -> Option<Self> {
        let is_digit = |r: &PosRun| !run_text(arena, r).trim().is_empty();
        let is_sign = |r: &PosRun| r.t0 == r.t1;

        let mut rows = cluster_rows(body);
        if rows.is_empty() {
            return None;
        }
        let digits = body.iter().filter(|r| is_digit(r)).count();
        let n_rows = rows.len();
        if digits == 0 || digits % n_rows != 0 {
            return None;
        }
        let cols = digits / n_rows;
        if cols == 0 {
            return None;
        }

        let mut grid: Vec<Vec<Cow<'a, str>>> = Vec::with_capacity(n_rows);
        for row in &mut rows {
            row.sort_by(|&a, &b| body[a].x.total_cmp(&body[b].x));
            let digit_idx: Vec<usize> = row
                .iter()
                .copied()
                .filter(|&k| body.get(k).is_some_and(&is_digit))
                .collect();
            if digit_idx.len() != cols {
                return None;
            }
            let mut cells: Vec<Cow<'a, str>> = digit_idx
                .iter()
                .map(|&k| Cow::Borrowed(run_text(arena, &body[k])))
                .collect();
            // A nothing-decoded glyph left of a digit is the matrix's lost minus:
            // attach it to the column of the first digit to its right.
            for &k in row.iter() {
                let Some(run) = body.get(k) else { continue };
                if !is_sign(run) {
                    continue;
                }
                let col = digit_idx.iter().filter(|&&d| body[d].x < run.x).count();
                if let Some(cell) = cells.get_mut(col) {
                    *cell = Cow::Owned(format!("-{cell}"));
                }
            }
            grid.push(cells);
        }
        Some(Self { delim, rows: grid })
    }

    /// Max display width of each column (cells are ASCII, so chars == columns).
    fn widths(&self) -> Vec<usize> {
        let cols = self.rows.first().map_or(0, Vec::len);
        (0..cols)
            .map(|c| {
                self.rows
                    .iter()
                    .filter_map(|row| row.get(c))
                    .map(|cell| cell.chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect()
    }
}

/// Cluster body-run indices into rows by Y, using an adaptive gap threshold so
/// the result is scale-independent. Returns indices into `body`, top row first.
fn cluster_rows(body: &[PosRun]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..body.len()).collect();
    order.sort_by(|&a, &b| body[a].y.total_cmp(&body[b].y));

    let mut large: Vec<f32> = order
        .windows(2)
        .map(|w| body[w[1]].y - body[w[0]].y)
        .filter(|g| *g > ROW_MIN_GAP)
        .collect();
    let tol = if large.is_empty() {
        f32::INFINITY
    } else {
        large.sort_by(f32::total_cmp);
        large[large.len() / 2] * 0.5
    };

    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut prev_y: Option<f32> = None;
    for &idx in &order {
        let y = body[idx].y;
        if prev_y.is_some_and(|p| y - p > tol) {
            rows.push(std::mem::take(&mut cur));
        }
        cur.push(idx);
        prev_y = Some(y);
    }
    if !cur.is_empty() {
        rows.push(cur);
    }
    rows
}

/// Unicode box-art rendering of the matrix (right-aligned columns).
impl fmt::Display for Matrix<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths = self.widths();
        let inner: usize = widths.iter().sum::<usize>() + widths.len().saturating_sub(1) * 2 + 2;
        let bar = " ".repeat(inner);
        writeln!(f, "┌{bar}┐")?;
        for row in &self.rows {
            f.write_str("│ ")?;
            for (c, cell) in row.iter().enumerate() {
                if c > 0 {
                    f.write_str("  ")?;
                }
                let w = widths.get(c).copied().unwrap_or(0);
                write!(f, "{cell:>w$}")?;
            }
            writeln!(f, " │")?;
        }
        write!(f, "└{bar}┘")
    }
}

/// LaTeX rendering: a `$$ \begin{pmatrix} … \end{pmatrix} $$` block.
struct Latex<'a>(&'a Matrix<'a>);

impl fmt::Display for Latex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let env = self.0.delim.env();
        writeln!(f, "$$")?;
        writeln!(f, "\\begin{{{env}}}")?;
        let n = self.0.rows.len();
        for (i, row) in self.0.rows.iter().enumerate() {
            f.write_str(&row.join(" & "))?;
            if i + 1 < n {
                f.write_str(" \\\\")?;
            }
            f.write_char('\n')?;
        }
        writeln!(f, "\\end{{{env}}}")?;
        write!(f, "$$")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::{GlyphDecoder, interpret_runs};

    /// Identity (WinAnsi) decoder so a raw content stream drives the full path.
    struct Win;
    impl GlyphDecoder for Win {
        fn decode(&self, _font: Option<&[u8]>, bytes: &[u8], out: &mut String) {
            crate::encoding::push_winansi(out, bytes);
        }
    }

    #[test]
    fn pipeline_from_content_stream() {
        // A paren-fenced 2×2 matrix positioned glyph-by-glyph via `Tm`:
        //   ( 1 2 )
        //   ( 3 4 )
        // left fence `0`/`@` at x=0, right fence `1`/`A` at x=30.
        let cs = b"BT /F1 10 Tf \
            1 0 0 1 0 100 Tm (0) Tj 1 0 0 1 0 110 Tm (@) Tj \
            1 0 0 1 10 100 Tm (1) Tj 1 0 0 1 20 100 Tm (2) Tj \
            1 0 0 1 10 110 Tm (3) Tj 1 0 0 1 20 110 Tm (4) Tj \
            1 0 0 1 30 100 Tm (1) Tj 1 0 0 1 30 110 Tm (A) Tj ET";
        let page = interpret_runs(cs, &Win);
        let lines = rewrite(&page);
        let text: String = lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("\\begin{pmatrix}"), "latex: {text}");
        assert!(text.contains("1 & 2"), "row0: {text}");
        assert!(text.contains("3 & 4"), "row1: {text}");
        assert!(!text.contains("0BBBBB@"), "no raw delimiter junk: {text}");
    }

    /// Build a `PosRun` for a single glyph at (x, y); `text` "" ⇒ empty decode.
    fn run(arena: &mut String, x: f32, y: f32, text: &str, nl: bool) -> PosRun {
        let t0 = arena.len() as u32;
        arena.push_str(text);
        let t1 = arena.len() as u32;
        PosRun {
            x,
            y,
            size: 10.0,
            t0,
            t1,
            nl,
        }
    }

    /// A 6-wide paren fence stack (`0 B B B B B @` / `1 C C C C C A`) at `x`.
    fn fence(arena: &mut String, runs: &mut Vec<PosRun>, x: f32, glyphs: &[&str]) {
        for (k, g) in glyphs.iter().enumerate() {
            // Y increases downward; top hook first.
            runs.push(run(arena, x, 100.0 + k as f32 * 10.0, g, false));
        }
    }

    #[test]
    fn reconstructs_signed_two_by_two() {
        // ( 1  -1 )
        // ( 0   2 )
        let mut arena = String::new();
        let mut runs: Vec<PosRun> = Vec::new();
        fence(&mut arena, &mut runs, 0.0, &["0", "@"]);
        // row 0 (y=100): 1 at x=10, then minus(empty) at x=18 + 1 at x=22
        runs.push(run(&mut arena, 10.0, 100.0, "1", false));
        runs.push(run(&mut arena, 18.0, 100.0, "", false)); // lost minus
        runs.push(run(&mut arena, 22.0, 100.0, "1", false));
        // row 1 (y=112): 0 at x=10, 2 at x=22
        runs.push(run(&mut arena, 10.0, 112.0, "0", false));
        runs.push(run(&mut arena, 22.0, 112.0, "2", false));
        fence(&mut arena, &mut runs, 40.0, &["1", "A"]);

        let regions = detect(&runs, &arena);
        assert_eq!(regions.len(), 1, "one matrix region");
        let block = &regions[0].block;
        assert!(block.contains("\\begin{pmatrix}"), "latex env: {block}");
        assert!(block.contains("1 & -1"), "row0 with recovered sign: {block}");
        assert!(block.contains("0 & 2"), "row1: {block}");
        assert!(block.contains('┌') && block.contains('┘'), "box-art: {block}");
    }

    #[test]
    fn reconstructs_column_vector() {
        // ( 1 )
        // ( 1 )
        // ( 1 )
        let mut arena = String::new();
        let mut runs: Vec<PosRun> = Vec::new();
        fence(&mut arena, &mut runs, 0.0, &["0", "@"]);
        runs.push(run(&mut arena, 10.0, 100.0, "1", false));
        runs.push(run(&mut arena, 10.0, 112.0, "1", false));
        runs.push(run(&mut arena, 10.0, 124.0, "1", false));
        fence(&mut arena, &mut runs, 20.0, &["1", "A"]);

        let regions = detect(&runs, &arena);
        assert_eq!(regions.len(), 1);
        let block = &regions[0].block;
        assert_eq!(block.matches("\\\\").count(), 2, "3 rows ⇒ 2 row separators: {block}");
    }

    #[test]
    fn plain_text_is_left_untouched() {
        // Runs that are not a fence produce no regions, so assemble passes the
        // lines through unchanged.
        let mut arena = String::new();
        let hello = run(&mut arena, 0.0, 100.0, "Hello", false);
        let world = run(&mut arena, 40.0, 100.0, "world", false);
        let runs = vec![hello, world];
        let regions = detect(&runs, &arena);
        assert!(regions.is_empty());
        let page = RunPage { runs, arena };
        let lines = rewrite(&page);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello world");
    }
}
