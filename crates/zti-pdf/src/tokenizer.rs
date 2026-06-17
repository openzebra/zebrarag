//! Low-level PDF content-stream tokeniser primitives.
//!
//! These functions turn raw content-stream bytes into operands (strings,
//! numbers, names, array markers) without any interpretation of what the
//! operands *mean*. The [`crate::interpreter`] layer consumes them.

/// Operands pushed between operators. Strings are materialised (escapes/hex
/// must be decoded, so they cannot borrow); a name carries its raw bytes so
/// `Tf` can resolve the current font's resource name for glyph decoding.
#[derive(Debug, Clone)]
pub enum Operand {
    Str(Vec<u8>),
    Name(Vec<u8>),
    Num(f32),
    ArrayStart,
}

/// Whitespace test for the PDF tokeniser (NUL/TAB/LF/FF/CR/SPACE).
#[inline]
pub const fn is_ws(b: u8) -> bool {
    matches!(b, 0x00 | 0x09 | 0x0A | 0x0C | 0x0D | 0x20)
}

/// Delimiter + whitespace test for the "regular character" run consumed by a
/// bare operator / number / name-body token.
#[inline]
pub const fn is_regular(b: u8) -> bool {
    !matches!(
        b,
        0x00 | 0x09 | 0x0A | 0x0C | 0x0D | 0x20 | // whitespace
        0x28 | 0x29 | 0x3C | 0x3E | 0x5B | 0x5D | 0x7B | 0x7D | // ( ) < > [ ] { }
        0x2F | 0x25 // / %
    )
}

/// Hex digit value, or `None` for non-hex bytes.
#[inline]
pub const fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse a PDF literal string starting just after the opening `(`. Returns the
/// decoded byte payload and the index just past the closing `)`. Handles
/// backslash escapes, nested unescaped parens, and `\ddd` octal byte values.
pub fn parse_literal(data: &[u8], start: usize) -> (Vec<u8>, usize) {
    let mut out = Vec::new();
    let mut i = start;
    let mut depth: u32 = 1;
    while i < data.len() {
        let b = data[i];
        match b {
            b'\\' => {
                i += 1;
                let Some(&e) = data.get(i) else { break };
                match e {
                    b'n' => {
                        out.push(0x0A);
                        i += 1;
                    }
                    b'r' => {
                        out.push(0x0D);
                        i += 1;
                    }
                    b't' => {
                        out.push(0x09);
                        i += 1;
                    }
                    b'b' => {
                        out.push(0x08);
                        i += 1;
                    }
                    b'f' => {
                        out.push(0x0C);
                        i += 1;
                    }
                    b'(' => {
                        out.push(b'(');
                        i += 1;
                    }
                    b')' => {
                        out.push(b')');
                        i += 1;
                    }
                    b'\\' => {
                        out.push(b'\\');
                        i += 1;
                    }
                    b'\n' => {
                        i += 1;
                    }
                    b'\r' => {
                        i += 1;
                        if data.get(i) == Some(&b'\n') {
                            i += 1;
                        }
                    }
                    d if d.is_ascii_digit() => {
                        // Up to three octal digits.
                        let mut val: u32 = 0;
                        let mut k = 0;
                        while k < 3
                            && let Some(&dd) = data.get(i)
                            && dd.is_ascii_digit()
                        {
                            val = val * 8 + u32::from(dd - b'0');
                            i += 1;
                            k += 1;
                        }
                        if let Ok(byte) = u8::try_from(val) {
                            out.push(byte);
                        }
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            b'(' => {
                out.push(b'(');
                depth = depth.saturating_add(1);
                i += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return (out, i + 1);
                }
                out.push(b')');
                i += 1;
            }
            _ => {
                out.push(b);
                i += 1;
            }
        }
    }
    (out, i)
}

/// Parse a PDF hex string starting just after the opening `<`. Returns the
/// decoded bytes (high nibble first, odd trailing nibble zero-padded) and the
/// index just past the closing `>`.
pub fn parse_hex(data: &[u8], start: usize) -> (Vec<u8>, usize) {
    let mut out = Vec::new();
    let mut i = start;
    let mut hi: Option<u8> = None;
    while i < data.len() {
        let b = data[i];
        if b == b'>' {
            if let Some(h) = hi {
                out.push(h.wrapping_shl(4));
            }
            return (out, i + 1);
        }
        if let Some(d) = hex_val(b) {
            match hi {
                None => hi = Some(d),
                Some(h) => {
                    out.push(h.wrapping_shl(4) | d);
                    hi = None;
                }
            }
        }
        i += 1;
    }
    (out, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_string_unescapes_parens() {
        // Input bytes: f o o \( b a r \( b a z \ ) )
        // `\(` and `\)` are escaped parens (output verbatim); the final
        // unescaped `)` closes the literal, so it does NOT appear in output.
        let input = b"foo\\(bar\\(baz\\))";
        let (s, next) = parse_literal(input, 0);
        assert_eq!(s, b"foo(bar(baz)");
        assert_eq!(next, input.len());
    }

    #[test]
    fn literal_string_octal_escape() {
        let (s, _) = parse_literal(b"\\101\\102\\103)", 0);
        assert_eq!(s, b"ABC");
    }

    #[test]
    fn literal_string_handles_nested_unescaped_parens() {
        let (s, next) = parse_literal(b"a(b)c)", 0);
        assert_eq!(s, b"a(b)c");
        assert_eq!(next, b"a(b)c)".len());
    }

    #[test]
    fn hex_string_decodes_pairs() {
        let (s, next) = parse_hex(b"48656C6C6F>", 0);
        assert_eq!(s, b"Hello");
        assert_eq!(next, b"48656C6C6F>".len());
    }

    #[test]
    fn hex_string_odd_nibble_zero_pads() {
        let (s, _) = parse_hex(b"4>", 0);
        assert_eq!(s, b"@"); // 0x40
    }

    #[test]
    fn hex_string_ignores_whitespace() {
        let (s, _) = parse_hex(b"48 65>", 0);
        assert_eq!(s, b"He");
    }

    #[test]
    fn hex_val_boundaries() {
        assert_eq!(hex_val(b'0'), Some(0));
        assert_eq!(hex_val(b'9'), Some(9));
        assert_eq!(hex_val(b'a'), Some(10));
        assert_eq!(hex_val(b'F'), Some(15));
        assert_eq!(hex_val(b'g'), None);
        assert_eq!(hex_val(b' '), None);
    }
}
