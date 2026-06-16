//! PDF content-stream text interpreter.
//!
//! Walks one page's content stream into a flat list of positioned runs
//! ([`interpret_runs`]) and folds those runs back into rendered [`Line`]s
//! ([`assemble`]). Splitting the two lets [`crate::mathrec`] inspect the glyph
//! geometry (which the flattened lines discard) and splice reconstructed math
//! blocks in by run index without losing the prose around them.
//!
//! Line breaks are derived from the positioning operators (`Td`, `TD`, `T*`,
//! `Tm`) via a Y-axis-drop heuristic relative to the current font size.
//! Inter-word spaces are derived from horizontal repositioning between
//! text-showing ops: when the X position advances beyond a quarter of the font
//! size since the last glyph, a space is inserted. Only the text-showing subset
//! of the content stream is interpreted; graphics and colour operators fall
//! through to the operand stack and are cleared on the next consumed operator.

use crate::tokenizer::{Operand, is_regular, is_ws, parse_hex, parse_literal};

/// One rendered line of page text plus the font size active for its glyphs.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub text: String,
    pub font_size: f32,
}

/// One positioned text run: the glyphs shown at a single (x, y) under one font
/// size, with the decoded characters held as a `t0..t1` slice of the page arena
/// (an empty slice ⇒ the glyph decoded to nothing, e.g. an obfuscated minus).
/// `nl` records that a line break preceded this run (operator-driven, so it is
/// exact rather than re-derived from the Y delta). This is the geometry the
/// flattened [`Line`] stream throws away, retained for [`crate::mathrec`].
#[derive(Debug, Clone, PartialEq)]
pub struct PosRun {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub t0: u32,
    pub t1: u32,
    pub nl: bool,
}

/// One page's positioned runs plus the character arena their `t0..t1` index.
#[derive(Debug, Default)]
pub struct RunPage {
    pub runs: Vec<PosRun>,
    pub arena: String,
}

/// A reconstructed math block occupying runs `run_lo..=run_hi`. [`assemble`]
/// drops those runs and emits `block` as a single [`Line`] (with `size` chosen
/// so heading detection never mistakes it for a heading).
#[derive(Debug)]
pub struct Region {
    pub run_lo: usize,
    pub run_hi: usize,
    pub size: f32,
    pub block: String,
}

/// Decodes a text-showing operand's raw bytes into Unicode under the font named
/// by the current `Tf`. Implemented in [`crate::extract`] over the PDF's
/// per-font encodings (ToUnicode CMap / `Differences` / base encoding); the
/// interpreter stays free of any PDF-object knowledge via static dispatch.
pub trait GlyphDecoder {
    /// Append the decoded glyphs of `bytes` to `out`. `font` is the current font
    /// resource name (the `/Fn` operand of `Tf`), or `None` before any `Tf`.
    fn decode(&self, font: Option<&[u8]>, bytes: &[u8], out: &mut String);
}

/// Font-oblivious decoder: every byte maps through WinAnsiEncoding. The
/// production fallback lives in [`crate::extract`]; this is the Tier-1 stand-in
/// the interpreter's own unit tests drive directly.
#[cfg(test)]
struct WinAnsi;

#[cfg(test)]
impl GlyphDecoder for WinAnsi {
    fn decode(&self, _font: Option<&[u8]>, bytes: &[u8], out: &mut String) {
        crate::encoding::push_winansi(out, bytes);
    }
}

/// Interpret one page's content stream into rendered text lines, decoding shown
/// text with `decoder` (font-aware). Convenience composition of
/// [`interpret_runs`] and [`assemble`] with no math reconstruction — used by the
/// interpreter's own tests; production goes through `interpret_runs` +
/// [`crate::mathrec::rewrite`].
#[cfg(test)]
fn interpret(data: &[u8], decoder: &impl GlyphDecoder) -> Vec<Line> {
    let page = interpret_runs(data, decoder);
    assemble(&page.runs, &page.arena, Vec::new())
}

/// Mutable text state for one content-stream walk. Holds only positioning and
/// the decoded-glyph arena; line/space/script rendering is deferred to
/// [`assemble`] so it can be shared with math reconstruction.
#[derive(Default)]
struct Interp {
    runs: Vec<PosRun>,
    arena: String,
    size: f32,
    y: f32,
    last_line_y: f32,
    lead: f32,
    in_text: bool,
    x: f32,
    line_x: f32,
    font: Vec<u8>,
    pending_nl: bool,
}

impl Interp {
    /// Emit one positioned run for the shown `bytes`, decoding into the arena.
    fn emit(&mut self, bytes: &[u8], decoder: &impl GlyphDecoder) {
        let font = (!self.font.is_empty()).then_some(self.font.as_slice());
        let t0 = self.arena.len() as u32;
        decoder.decode(font, bytes, &mut self.arena);
        let t1 = self.arena.len() as u32;
        self.runs.push(PosRun {
            x: self.x,
            y: self.y,
            size: self.size,
            t0,
            t1,
            nl: self.pending_nl,
        });
        self.pending_nl = false;
    }

    /// Apply one content-stream operator, updating text state and emitting runs.
    fn op(&mut self, op: &str, stack: &[Operand], decoder: &impl GlyphDecoder) {
        let n = stack.len();
        match op {
            "BT" => {
                self.in_text = true;
                self.y = 0.0;
                self.last_line_y = 0.0;
                self.x = 0.0;
                self.line_x = 0.0;
            }
            "ET" => {
                self.in_text = false;
                self.pending_nl = true;
            }
            "Tf" if n >= 2 => {
                if let Some(Operand::Num(size)) = stack.get(n - 1) {
                    self.size = *size;
                }
                // Operands are `/Fn size Tf`; remember the resource name so shown
                // text decodes under the right font's encoding.
                if let Some(Operand::Name(name)) = stack.get(n - 2) {
                    self.font.clear();
                    self.font.extend_from_slice(name);
                }
            }
            "Tm" if n >= 6 => {
                // operands: a b c d e f  → position (e, f)
                if let (Some(Operand::Num(e)), Some(Operand::Num(f))) =
                    (stack.get(n - 2), stack.get(n - 1))
                {
                    if self.in_text && *f < self.last_line_y - self.size * 0.3 {
                        self.pending_nl = true;
                    }
                    self.x = *e;
                    self.line_x = *e;
                    self.y = *f;
                    self.last_line_y = *f;
                }
            }
            "Td" if n >= 2 => {
                if let (Some(Operand::Num(tx)), Some(Operand::Num(ty))) =
                    (stack.get(n - 2), stack.get(n - 1))
                {
                    if self.in_text && *ty < -self.size * 0.3 {
                        self.pending_nl = true;
                    }
                    // Td translates the text line matrix by (tx, ty); the new
                    // absolute X is the accumulated line-start plus tx.
                    self.line_x += *tx;
                    self.x = self.line_x;
                    self.y += *ty;
                    self.last_line_y = self.y;
                }
            }
            "TD" if n >= 2 => {
                if let (Some(Operand::Num(tx)), Some(Operand::Num(ty))) =
                    (stack.get(n - 2), stack.get(n - 1))
                {
                    self.lead = -*ty;
                    if self.in_text && *ty < -self.size * 0.3 {
                        self.pending_nl = true;
                    }
                    self.line_x += *tx;
                    self.x = self.line_x;
                    self.y += *ty;
                    self.last_line_y = self.y;
                }
            }
            "TL" if n >= 1 => {
                if let Some(Operand::Num(l)) = stack.get(n - 1) {
                    self.lead = *l;
                }
            }
            "T*" if self.in_text => {
                self.pending_nl = true;
                // T* is equivalent to `0 -lead Td`: X stays at the line start.
                self.x = self.line_x;
                self.y -= self.lead;
                self.last_line_y = self.y;
            }
            "Tj" | "TJ" if n >= 1 && self.in_text => {
                if let Some(Operand::Str(s)) = stack.get(n - 1) {
                    self.emit(s, decoder);
                }
            }
            "'" if n >= 1 && self.in_text => {
                self.pending_nl = true;
                self.y -= self.lead;
                self.last_line_y = self.y;
                if let Some(Operand::Str(s)) = stack.get(n - 1) {
                    self.emit(s, decoder);
                }
            }
            "\"" if n >= 3 && self.in_text => {
                self.pending_nl = true;
                self.y -= self.lead;
                self.last_line_y = self.y;
                if let Some(Operand::Str(s)) = stack.get(n - 1) {
                    self.emit(s, decoder);
                }
            }
            _ => {}
        }
    }
}

/// Walk one page's content stream into positioned runs without rendering lines.
pub fn interpret_runs(data: &[u8], decoder: &impl GlyphDecoder) -> RunPage {
    let mut it = Interp::default();
    let mut stack: Vec<Operand> = Vec::with_capacity(16);

    let mut i = 0;
    while let Some(&b0) = data.get(i) {
        match b0 {
            b if is_ws(b) => {
                i += 1;
            }
            b'%' => {
                // Comment runs to end of line.
                while let Some(&c) = data.get(i) {
                    i += 1;
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                }
            }
            b'(' => {
                let (s, next) = parse_literal(data, i + 1);
                stack.push(Operand::Str(s));
                i = next;
            }
            b'<' => {
                if data.get(i + 1) == Some(&b'<') {
                    // Dict start — Tier-1 ignores inline resource dicts; just
                    // consume the `<<` so it isn't seen as a hex string.
                    i += 2;
                } else {
                    let (s, next) = parse_hex(data, i + 1);
                    stack.push(Operand::Str(s));
                    i = next;
                }
            }
            b'>' if data.get(i + 1) == Some(&b'>') => {
                // Dict end — consume `>>`.
                i += 2;
            }
            b'[' => {
                stack.push(Operand::ArrayStart);
                i += 1;
            }
            b']' => {
                // Collapse everything since the matching `[` into one Str (the
                // TJ array form: `[ (a) -10 (b) ... ]`). Numbers act as kerning
                // deltas and are dropped; only string bytes survive.
                let mut parts: Vec<Vec<u8>> = Vec::new();
                while let Some(op) = stack.pop() {
                    match op {
                        Operand::ArrayStart => break,
                        Operand::Str(s) => parts.push(s),
                        Operand::Num(_) | Operand::Name(_) => {}
                    }
                }
                let total: usize = parts.iter().map(Vec::len).sum();
                let mut concat = Vec::with_capacity(total);
                for p in parts.iter().rev() {
                    concat.extend_from_slice(p);
                }
                stack.push(Operand::Str(concat));
                i += 1;
            }
            b'/' => {
                let mut j = i + 1;
                while let Some(&c) = data.get(j) {
                    if is_regular(c) {
                        j += 1;
                    } else {
                        break;
                    }
                }
                // Carry the name body (between `/` and the delimiter) so `Tf`
                // can resolve the font resource name.
                if let Some(name) = data.get(i + 1..j) {
                    stack.push(Operand::Name(name.to_vec()));
                }
                i = j;
            }
            _ => {
                let start = i;
                while let Some(&c) = data.get(i) {
                    if is_regular(c) {
                        i += 1;
                    } else {
                        break;
                    }
                }
                if i == start {
                    // Stray delimiter we don't model (e.g. `{`, `}`); skip.
                    i += 1;
                    continue;
                }
                let Some(slice) = data.get(start..i) else {
                    stack.clear();
                    continue;
                };
                let Ok(s) = std::str::from_utf8(slice) else {
                    stack.clear();
                    continue;
                };
                if let Ok(num) = s.parse::<f32>() {
                    stack.push(Operand::Num(num));
                } else {
                    it.op(s, &stack, decoder);
                    stack.clear();
                }
            }
        }
    }

    RunPage {
        runs: it.runs,
        arena: it.arena,
    }
}

/// Fold positioned `runs` (indexing `arena`) into rendered [`Line`]s, splicing
/// each [`Region`] in as a single block line in place of its runs. With an empty
/// `regions` this is the exact inverse of [`interpret_runs`] — the same line,
/// inter-word-space and sub/superscript rendering the flat interpreter produced.
pub fn assemble(runs: &[PosRun], arena: &str, regions: Vec<Region>) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::with_capacity(runs.len() / 8 + regions.len() + 1);
    let mut cur = String::new();
    let mut line_size = 0.0f32;
    // Sub/superscript baseline reference for the current line, reset on break.
    let mut base_size = 0.0f32;
    let mut base_y = 0.0f32;
    let mut script: i8 = 0;
    let mut last_text_x = 0.0f32;
    let mut regions = regions.into_iter().peekable();

    let mut i = 0usize;
    while i < runs.len() {
        if regions.peek().is_some_and(|r| r.run_lo == i) {
            if let Some(reg) = regions.next() {
                flush(&mut out, &mut cur, line_size);
                out.push(Line {
                    text: reg.block,
                    font_size: reg.size,
                });
                i = reg.run_hi + 1;
                base_size = 0.0;
                base_y = 0.0;
                script = 0;
            }
            continue;
        }
        let Some(run) = runs.get(i) else { break };
        if run.nl {
            flush(&mut out, &mut cur, line_size);
            base_size = 0.0;
            base_y = 0.0;
            script = 0;
        }
        // Classify body / subscript / superscript exactly as the flat path did.
        let smaller = base_size > 0.0 && run.size < base_size * 0.8;
        let dy = run.y - base_y;
        let kind: i8 = if smaller && dy < -base_size * 0.05 {
            -1
        } else if smaller && dy > base_size * 0.05 {
            1
        } else {
            0
        };
        if kind == 0 {
            base_size = run.size;
            base_y = run.y;
            script = 0;
            if !cur.is_empty() && run.x > last_text_x + run.size * 0.25 {
                cur.push(' ');
            }
        } else if script != kind {
            cur.push(if kind < 0 { '_' } else { '^' });
            script = kind;
        }
        if let Some(slice) = arena.get(run.t0 as usize..run.t1 as usize) {
            push_glyphs(&mut cur, slice);
        }
        last_text_x = run.x;
        line_size = run.size;
        i += 1;
    }
    flush(&mut out, &mut cur, line_size);
    out
}

/// Push a finished line into `out`, taking ownership of the buffer's contents.
fn flush(out: &mut Vec<Line>, cur: &mut String, size: f32) {
    if !cur.is_empty() {
        out.push(Line {
            text: std::mem::take(cur),
            font_size: size,
        });
    }
}

/// Append decoded glyphs to `cur`, mapping control bytes to spaces and
/// collapsing repeated whitespace — line breaks come from positioning ops, not
/// embedded control bytes.
fn push_glyphs(cur: &mut String, s: &str) {
    cur.reserve(s.len());
    for c in s.chars() {
        let c = if c.is_control() { ' ' } else { c };
        if c == ' ' && cur.ends_with(' ') {
            continue;
        }
        cur.push(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpret_simple_tj() {
        // BT /F1 12 Tf 100 700 Td (Hello) Tj ET
        let cs = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello");
        assert_eq!(lines[0].font_size, 12.0);
    }

    #[test]
    fn interpret_tj_concatenates_until_newline() {
        let cs = b"BT /F1 10 Tf (Hello) Tj ( ) Tj (World) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello World");
    }

    #[test]
    fn interpret_td_downward_starts_new_line() {
        // Two Td moves that drop Y → two lines.
        let cs = b"BT /F1 10 Tf 0 700 Td (One) Tj 0 -12 Td (Two) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "One");
        assert_eq!(lines[1].text, "Two");
    }

    #[test]
    fn interpret_tj_array_collapses_kerning() {
        let cs = b"BT /F1 10 Tf [(Hel)-30(lo)] TJ ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello");
    }

    #[test]
    fn interpret_t_star_newline() {
        let cs = b"BT /F1 10 Tf 12 TL (A) Tj T* (B) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "A");
        assert_eq!(lines[1].text, "B");
    }

    #[test]
    fn interpret_outside_text_block_is_dropped() {
        let cs = b"(Ghost) Tj BT /F1 10 Tf (Real) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Real");
    }

    #[test]
    fn interpret_marks_subscript_with_underscore() {
        // `a` at 10pt baseline, then `j` at 7pt shifted down ~2 units (a small
        // drop that does not trigger a line break): a subscript → `a_j`.
        let cs = b"BT /F1 10 Tf 0 700 Td (a) Tj /F2 7 Tf 5 -2 Td (j) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "a_j");
    }

    #[test]
    fn interpret_marks_superscript_with_caret() {
        // `x` at 10pt, then `2` at 7pt shifted up: a superscript → `x^2`.
        let cs = b"BT /F1 10 Tf 0 700 Td (x) Tj /F2 7 Tf 5 4 Td (2) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "x^2");
    }

    #[test]
    fn interpret_same_size_shift_is_not_a_script() {
        // Same font size with a tiny baseline jitter must stay body text — no
        // spurious `_`/`^` on ordinary prose.
        let cs = b"BT /F1 10 Tf 0 700 Td (a) Tj 5 -1 Td (b) Tj ET";
        let lines = interpret(cs, &WinAnsi);
        assert_eq!(lines.len(), 1);
        assert!(!lines[0].text.contains('_'));
        assert!(!lines[0].text.contains('^'));
    }
}
