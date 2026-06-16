//! Shared test utilities for `zti-pdf` integration tests.
//!
//! [`build_sample_pdf`] hand-authors a minimal, fully-valid 3-page PDF whose
//! content streams are designed to exercise the three PDF-extraction
//! regressions that Phase 2 of the indexing work must fix:
//!
//! 1. **Baseline** (page 1): a real heading (`Algorithm C` at 18 pt) above body
//!    text at 10 pt. Heading detection must succeed here.
//! 2. **Inter-word spacing** (page 2): two `Tj` ops with a horizontal `Td` gap
//!    and *no space byte* between them. Current extraction glues the words;
//!    the spec test requires a space.
//! 3. **Heading validation** (page 3): an 18 pt line that is mostly
//!    digits/punctuation (`x70.[M33]`) above body text. Current heading
//!    detection picks it up; the spec test requires it to be rejected.
//!
//! The bytes are constructed in Rust with computed `xref` offsets so the file
//! is a valid PDF that `lopdf` loads — no committed binary blob, no extra
//! dependencies.

/// Build the synthetic 3-page sample PDF.
pub fn build_sample_pdf() -> Vec<u8> {
    // Page 1: heading + body + example. Separate BT/ET blocks per font size so
    // the interpreter flushes each line with the font size active when it was
    // drawn (the flush fires on `ET` before `Tf` can change the size).
    let p1 = b"BT /F1 18 Tf 72 700 Td (Algorithm C) Tj ET\nBT /F1 10 Tf 72 680 Td (Permutation generation by cyclic shifts.) Tj 0 -14 Td (Example: 1234, 2341, 3412.) Tj ET";
    // Page 2: `15 0 Td` moves X but not Y, so the interpreter keeps both `Tj`
    // ops on one line and concatenates without a space.
    let p2 = b"BT /F1 10 Tf 72 700 Td (Permutation) Tj 15 0 Td (generation) Tj ET";
    // Page 3: 18 pt line is mostly digits/punctuation; body is 10 pt.
    let p3 = b"BT /F1 18 Tf 72 700 Td (x70.[M33]) Tj ET\nBT /F1 10 Tf 72 680 Td (Some real body text here.) Tj ET";

    let mut buf: Vec<u8> = Vec::with_capacity(2048);
    // 10 objects: catalog(1), pages(2), 3×page(3,5,7), 3×stream(4,6,8), font(9).
    let mut off = vec![0usize; 10];

    // `%` after the version marks the file as binary (per PDF spec) and keeps
    // tools happy.
    buf.extend_from_slice(b"%PDF-1.4%\n");

    // 1: Catalog.
    off[1] = buf.len();
    buf.extend_from_slice(b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n");

    // 2: Pages tree.
    off[2] = buf.len();
    buf.extend_from_slice(b"2 0 obj\n<</Type/Pages/Kids[3 0 R 5 0 R 7 0 R]/Count 3>>\nendobj\n");

    // Page object helper: identical resources, only /Contents differs.
    let write_page = |buf: &mut Vec<u8>, num: usize, contents: usize| {
        buf.extend_from_slice(
            format!(
                "{num} 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 9 0 R>>>>/Contents {contents} 0 R>>\nendobj\n"
            )
            .as_bytes(),
        );
    };
    // Content stream helper: Length is the exact byte count of the payload.
    let write_stream = |buf: &mut Vec<u8>, num: usize, payload: &[u8]| {
        buf.extend_from_slice(format!("{num} 0 obj\n<</Length {}>>\nstream\n", payload.len()).as_bytes());
        buf.extend_from_slice(payload);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    // 3,4: page 1 + its content stream.
    off[3] = buf.len();
    write_page(&mut buf, 3, 4);
    off[4] = buf.len();
    write_stream(&mut buf, 4, p1);

    // 5,6: page 2 + its content stream.
    off[5] = buf.len();
    write_page(&mut buf, 5, 6);
    off[6] = buf.len();
    write_stream(&mut buf, 6, p2);

    // 7,8: page 3 + its content stream.
    off[7] = buf.len();
    write_page(&mut buf, 7, 8);
    off[8] = buf.len();
    write_stream(&mut buf, 8, p3);

    // 9: a Type1 font. The interpreter reads the `Tf` font size only; the font
    // program itself is never consulted, so Helvetica is a placeholder.
    off[9] = buf.len();
    buf.extend_from_slice(b"9 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n");

    // Cross-reference table. Each entry is exactly 20 bytes (SP/LF terminator).
    let xref_pos = buf.len();
    buf.extend_from_slice(b"xref\n0 10\n");
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for i in 1..10 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[i]).as_bytes());
    }

    // Trailer + startxref pointer.
    buf.extend_from_slice(b"trailer\n<</Size 10/Root 1 0 R>>\nstartxref\n");
    buf.extend_from_slice(format!("{xref_pos}\n").as_bytes());
    buf.extend_from_slice(b"%%EOF");

    buf
}
