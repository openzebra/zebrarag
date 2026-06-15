//! PDF content-stream text interpreter.
//!
//! Walks one page's content stream, maintains a tiny operand stack and text
//! state (current font size, Y position, leading), and emits [`Line`]s of
//! decoded text annotated with the font size active when the text was shown.
//! Line breaks are derived from the positioning operators (`Td`, `TD`, `T*`,
//! `Tm`) via a Y-axis-drop heuristic relative to the current font size.
//!
//! Only the text-showing subset of the content stream is interpreted; graphics
//! and colour operators fall through to the operand stack and are cleared on
//! the next consumed operator.

use crate::encoding::decode_byte;
use crate::tokenizer::{Operand, is_regular, is_ws, parse_hex, parse_literal};

/// One rendered line of page text plus the font size active for its glyphs.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub text: String,
    pub font_size: f32,
}

/// Interpret one page's content stream into rendered text lines.
pub fn interpret(data: &[u8]) -> Vec<Line> {
    let mut out: Vec<Line> = Vec::new();
    let mut cur_line = String::new();
    let mut cur_size: f32 = 0.0;
    let mut cur_y: f32 = 0.0;
    let mut last_line_y: f32 = 0.0;
    let mut lead: f32 = 0.0;
    let mut in_text = false;
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
                        Operand::Num(_) | Operand::Name => {}
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
                stack.push(Operand::Name);
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
) {
    let n = stack.len();
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
        }
        "ET" => {
            flush(cur_line, *cur_size, out);
            *in_text = false;
        }
        "Tf" if n >= 2 => {
            if let Operand::Num(size) = &stack[n - 1] {
                *cur_size = *size;
            }
        }
        "Tm" if n >= 6 => {
            // operands: a b c d e f  → position (e, f)
            if let (Operand::Num(_e), Operand::Num(f)) = (&stack[n - 2], &stack[n - 1]) {
                if *in_text && *f < *last_line_y - *cur_size * 0.3 {
                    flush(cur_line, *cur_size, out);
                }
                *cur_y = *f;
                *last_line_y = *f;
            }
        }
        "Td" if n >= 2 => {
            if let Operand::Num(ty) = &stack[n - 1] {
                if *in_text && *ty < -*cur_size * 0.3 {
                    flush(cur_line, *cur_size, out);
                }
                *cur_y += *ty;
                *last_line_y = *cur_y;
            }
        }
        "TD" if n >= 2 => {
            if let Operand::Num(ty) = &stack[n - 1] {
                *lead = -*ty;
                if *in_text && *ty < -*cur_size * 0.3 {
                    flush(cur_line, *cur_size, out);
                }
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
            }
            *cur_y -= *lead;
            *last_line_y = *cur_y;
        }
        "Tj" | "TJ" if n >= 1 => {
            if *in_text && let Operand::Str(s) = &stack[n - 1] {
                append_decoded(cur_line, s);
            }
        }
        "'" if n >= 1 && *in_text => {
            flush(cur_line, *cur_size, out);
            *cur_y -= *lead;
            *last_line_y = *cur_y;
            if let Operand::Str(s) = &stack[n - 1] {
                append_decoded(cur_line, s);
            }
        }
        "\"" if n >= 3 && *in_text => {
            flush(cur_line, *cur_size, out);
            *cur_y -= *lead;
            *last_line_y = *cur_y;
            if let Operand::Str(s) = &stack[n - 1] {
                append_decoded(cur_line, s);
            }
        }
        _ => {}
    }
}

/// Decode raw bytes under WinAnsi and append to the current line, collapsing
/// repeated whitespace (line breaks come from positioning ops, not embedded
/// control bytes).
fn append_decoded(cur: &mut String, bytes: &[u8]) {
    cur.reserve(bytes.len());
    for &b in bytes {
        let c = decode_byte(b);
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
        let lines = interpret(cs);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello");
        assert_eq!(lines[0].font_size, 12.0);
    }

    #[test]
    fn interpret_tj_concatenates_until_newline() {
        let cs = b"BT /F1 10 Tf (Hello) Tj ( ) Tj (World) Tj ET";
        let lines = interpret(cs);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello World");
    }

    #[test]
    fn interpret_td_downward_starts_new_line() {
        // Two Td moves that drop Y → two lines.
        let cs = b"BT /F1 10 Tf 0 700 Td (One) Tj 0 -12 Td (Two) Tj ET";
        let lines = interpret(cs);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "One");
        assert_eq!(lines[1].text, "Two");
    }

    #[test]
    fn interpret_tj_array_collapses_kerning() {
        let cs = b"BT /F1 10 Tf [(Hel)-30(lo)] TJ ET";
        let lines = interpret(cs);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello");
    }

    #[test]
    fn interpret_t_star_newline() {
        let cs = b"BT /F1 10 Tf 12 TL (A) Tj T* (B) Tj ET";
        let lines = interpret(cs);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "A");
        assert_eq!(lines[1].text, "B");
    }

    #[test]
    fn interpret_outside_text_block_is_dropped() {
        let cs = b"(Ghost) Tj BT /F1 10 Tf (Real) Tj ET";
        let lines = interpret(cs);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Real");
    }
}
