//! PDF content-stream text interpreter.
//!
//! Walks one page's content stream, maintains a tiny operand stack and text
//! state (current font size, X/Y position, leading), and emits [`Line`]s of
//! decoded text annotated with the font size active when the text was shown.
//! Line breaks are derived from the positioning operators (`Td`, `TD`, `T*`,
//! `Tm`) via a Y-axis-drop heuristic relative to the current font size.
//! Inter-word spaces are derived from horizontal `Td`/`Tm` repositioning
//! between text-showing ops: when the X position advances beyond a quarter of
//! the font size since the last glyph, a space is inserted.
//!
//! Only the text-showing subset of the content stream is interpreted; graphics
//! and colour operators fall through to the operand stack and are cleared on
//! the next consumed operator.

use crate::tokenizer::{Operand, is_regular, is_ws, parse_hex, parse_literal};

/// One rendered line of page text plus the font size active for its glyphs.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub text: String,
    pub font_size: f32,
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
/// text with `decoder` (font-aware) rather than a fixed byte map.
pub fn interpret(data: &[u8], decoder: &impl GlyphDecoder) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    let mut cur_line = String::new();
    let mut cur_size: f32 = 0.0;
    let mut cur_y: f32 = 0.0;
    let mut last_line_y: f32 = 0.0;
    let mut lead: f32 = 0.0;
    let mut in_text = false;
    let mut cur_x: f32 = 0.0;
    let mut line_x: f32 = 0.0;
    let mut last_text_x: f32 = 0.0;
    let mut cur_font: Vec<u8> = Vec::new();
    let mut decode_buf = String::new();
    // Sub/superscript tracking: the line's running body baseline + size, and
    // the current run's script state (-1 subscript, 0 body, +1 superscript).
    let mut base_size: f32 = 0.0;
    let mut base_y: f32 = 0.0;
    let mut script: i8 = 0;
    let mut stack: Vec<Operand> = Vec::with_capacity(16);

    let flush = |cur: &mut String, size: f32, out: &mut Vec<Line>| {
        if !cur.is_empty() {
            out.push(Line {
                text: std::mem::take(cur),
                font_size: size,
            });
        }
    };

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
                stack.push(Operand::Name(data[i + 1..j].to_vec()));
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
                let slice = &data[start..i];
                let Ok(s) = std::str::from_utf8(slice) else {
                    stack.clear();
                    continue;
                };
                if let Ok(n) = s.parse::<f32>() {
                    stack.push(Operand::Num(n));
                } else {
                    dispatch_op(
                        s,
                        &stack,
                        &mut out,
                        &mut cur_line,
                        &mut cur_size,
                        &mut cur_y,
                        &mut last_line_y,
                        &mut lead,
                        &mut in_text,
                        &mut cur_x,
                        &mut line_x,
                        &mut last_text_x,
                        &mut cur_font,
                        &mut decode_buf,
                        &mut base_size,
                        &mut base_y,
                        &mut script,
                        decoder,
                    );
                    stack.clear();
                }
            }
        }
    }

    flush(&mut cur_line, cur_size, &mut out);
    out
}

#[allow(clippy::too_many_arguments)]
fn dispatch_op(
    op: &str,
    stack: &[Operand],
    out: &mut Vec<Line>,
    cur_line: &mut String,
    cur_size: &mut f32,
    cur_y: &mut f32,
    last_line_y: &mut f32,
    lead: &mut f32,
    in_text: &mut bool,
    cur_x: &mut f32,
    line_x: &mut f32,
    last_text_x: &mut f32,
    cur_font: &mut Vec<u8>,
    decode_buf: &mut String,
    base_size: &mut f32,
    base_y: &mut f32,
    script: &mut i8,
    decoder: &impl GlyphDecoder,
) {
    let n = stack.len();
    // Reset the sub/superscript baseline reference: every line break starts a
    // fresh body baseline, so a script run never leaks across lines.
    let reset_script = |base_size: &mut f32, script: &mut i8| {
        *base_size = 0.0;
        *script = 0;
    };
    let flush = |cur: &mut String, size: f32, out: &mut Vec<Line>| {
        if !cur.is_empty() {
            out.push(Line {
                text: std::mem::take(cur),
                font_size: size,
            });
        }
    };
    match op {
        "BT" => {
            *in_text = true;
            *cur_y = 0.0;
            *last_line_y = 0.0;
            *cur_x = 0.0;
            *line_x = 0.0;
            reset_script(base_size, script);
        }
        "ET" => {
            flush(cur_line, *cur_size, out);
            *in_text = false;
        }
        "Tf" if n >= 2 => {
            if let Operand::Num(size) = &stack[n - 1] {
                *cur_size = *size;
            }
            // Operands are `/Fn size Tf`; remember the resource name so shown
            // text decodes under the right font's encoding.
            if let Operand::Name(name) = &stack[n - 2] {
                cur_font.clear();
                cur_font.extend_from_slice(name);
            }
        }
        "Tm" if n >= 6 => {
            // operands: a b c d e f  → position (e, f)
            if let (Operand::Num(e), Operand::Num(f)) = (&stack[n - 2], &stack[n - 1]) {
                if *in_text && *f < *last_line_y - *cur_size * 0.3 {
                    flush(cur_line, *cur_size, out);
                    reset_script(base_size, script);
                }
                *cur_x = *e;
                *line_x = *e;
                *cur_y = *f;
                *last_line_y = *f;
            }
        }
        "Td" if n >= 2 => {
            if let (Operand::Num(tx), Operand::Num(ty)) = (&stack[n - 2], &stack[n - 1]) {
                if *in_text && *ty < -*cur_size * 0.3 {
                    flush(cur_line, *cur_size, out);
                    reset_script(base_size, script);
                }
                // Td translates the text line matrix by (tx, ty); the new
                // absolute X is the accumulated line-start plus tx.
                *line_x += *tx;
                *cur_x = *line_x;
                *cur_y += *ty;
                *last_line_y = *cur_y;
            }
        }
        "TD" if n >= 2 => {
            if let (Operand::Num(tx), Operand::Num(ty)) = (&stack[n - 2], &stack[n - 1]) {
                *lead = -*ty;
                if *in_text && *ty < -*cur_size * 0.3 {
                    flush(cur_line, *cur_size, out);
                    reset_script(base_size, script);
                }
                *line_x += *tx;
                *cur_x = *line_x;
                *cur_y += *ty;
                *last_line_y = *cur_y;
            }
        }
        "TL" if n >= 1 => {
            if let Operand::Num(l) = &stack[n - 1] {
                *lead = *l;
            }
        }
        "T*" => {
            if *in_text {
                flush(cur_line, *cur_size, out);
                reset_script(base_size, script);
            }
            // T* is equivalent to `0 -lead Td`: X stays at the line start.
            *cur_x = *line_x;
            *cur_y -= *lead;
            *last_line_y = *cur_y;
        }
        "Tj" | "TJ" if n >= 1 => {
            if *in_text && let Operand::Str(s) = &stack[n - 1] {
                show_run(
                    cur_line, s, *cur_x, *last_text_x, *cur_y, *cur_size, base_size, base_y,
                    script, cur_font, decode_buf, decoder,
                );
                *last_text_x = *cur_x;
            }
        }
        "'" if n >= 1 && *in_text => {
            flush(cur_line, *cur_size, out);
            reset_script(base_size, script);
            *cur_y -= *lead;
            *last_line_y = *cur_y;
            if let Operand::Str(s) = &stack[n - 1] {
                show_run(
                    cur_line, s, *cur_x, *last_text_x, *cur_y, *cur_size, base_size, base_y,
                    script, cur_font, decode_buf, decoder,
                );
                *last_text_x = *cur_x;
            }
        }
        "\"" if n >= 3 && *in_text => {
            flush(cur_line, *cur_size, out);
            reset_script(base_size, script);
            *cur_y -= *lead;
            *last_line_y = *cur_y;
            if let Operand::Str(s) = &stack[n - 1] {
                show_run(
                    cur_line, s, *cur_x, *last_text_x, *cur_y, *cur_size, base_size, base_y,
                    script, cur_font, decode_buf, decoder,
                );
                *last_text_x = *cur_x;
            }
        }
        _ => {}
    }
}

/// Show one text run, classifying it as body / subscript / superscript by
/// comparing its font size and baseline to the line's running body text. Body
/// runs reset the baseline and take an inter-word space on a wide X gap; script
/// runs emit a single `_`/`^` marker at the transition and never a space.
#[allow(clippy::too_many_arguments)]
fn show_run(
    cur: &mut String,
    bytes: &[u8],
    cur_x: f32,
    last_text_x: f32,
    cur_y: f32,
    cur_size: f32,
    base_size: &mut f32,
    base_y: &mut f32,
    script: &mut i8,
    cur_font: &[u8],
    decode_buf: &mut String,
    decoder: &impl GlyphDecoder,
) {
    let smaller = *base_size > 0.0 && cur_size < *base_size * 0.8;
    let dy = cur_y - *base_y;
    let kind: i8 = if smaller && dy < -*base_size * 0.05 {
        -1
    } else if smaller && dy > *base_size * 0.05 {
        1
    } else {
        0
    };

    if kind == 0 {
        *base_size = cur_size;
        *base_y = cur_y;
        *script = 0;
        if !cur.is_empty() && cur_x > last_text_x + cur_size * 0.25 {
            cur.push(' ');
        }
    } else if *script != kind {
        cur.push(if kind < 0 { '_' } else { '^' });
        *script = kind;
    }
    show_text(cur, bytes, cur_font, decode_buf, decoder);
}

/// Decode `bytes` under the current font (via `decoder`, reusing `decode_buf` as
/// scratch) and append to `cur`, collapsing repeated whitespace — line breaks
/// come from positioning ops, not embedded control bytes.
fn show_text(
    cur: &mut String,
    bytes: &[u8],
    cur_font: &[u8],
    decode_buf: &mut String,
    decoder: &impl GlyphDecoder,
) {
    let font = (!cur_font.is_empty()).then_some(cur_font);
    decode_buf.clear();
    decoder.decode(font, bytes, decode_buf);
    cur.reserve(decode_buf.len());
    for c in decode_buf.chars() {
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
