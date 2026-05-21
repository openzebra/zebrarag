use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use zti_common::line_byte_range;
use zti_tree_sitter::Language;
use zti_ts_core::types::Kind;

use crate::model::ProjectIndex;
use crate::render::dsl::{AST_HEADER, lang_label, lang_legend, load_manifest_content};

pub fn find_manifest(root: &Path) -> Option<String> {
    crate::index::MANIFEST_NAMES.iter().find_map(|name| {
        let p = root.join(name);
        std::fs::read_to_string(&p).ok()
    })
}

/// Writes the project-map-style preamble (AST_HEADER + all manifests +
/// per-language section headers) into `out`. Used once at the top of
/// `zebra-dsl chunks` output so the preamble is not duplicated per chunk.
pub fn write_preamble(index: &ProjectIndex, out: &mut String) {
    out.push_str(AST_HEADER);
    out.push('\n');

    for rel in &index.manifest_paths {
        if let Some(content) = load_manifest_content(&index.root, rel) {
            let _ = writeln!(out, "@ {}\n{}", rel, content);
            out.push('\n');
        }
    }

    let mut by_label: BTreeMap<&'static str, Language> = BTreeMap::new();
    for file in &index.files {
        by_label.entry(lang_label(file.language)).or_insert(file.language);
    }
    for (label, lang) in &by_label {
        let _ = writeln!(out, "## {}", label);
        out.push_str(lang_legend(*lang));
        out.push('\n');
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub file: String,
    pub rel_file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub sym_id: u32,
    pub body: String,
    pub qualified: String,
    pub kind: Kind,
}

impl Chunk {
    pub const DOCUMENT_SYM_ID: u32 = u32::MAX;

    pub fn is_document(&self) -> bool {
        self.sym_id == Self::DOCUMENT_SYM_ID
    }

    /// Text used by the embedding model and stored in `chunks.content`.
    /// One `String::with_capacity` allocation, no `format!` intermediate.
    pub fn embed_text(&self) -> Cow<'_, str> {
        let mut out = String::with_capacity(self.rel_file.len() + self.body.len() + 32);
        let _ = writeln!(out, "FILE: {} :{}-{}", self.rel_file, self.start_line, self.end_line);
        out.push_str(&self.body);
        Cow::Owned(out)
    }

    /// Same as `embed_text` (the body already contains the kind#id tag on the
    /// signature line and any preceding doc comments). Kept as a distinct
    /// method so the CLI display path can diverge later without touching the
    /// embedding format.
    pub fn display_text(&self) -> Cow<'_, str> {
        self.embed_text()
    }
}

pub struct DslChunker<'a> {
    index: &'a ProjectIndex,
}

impl<'a> DslChunker<'a> {
    pub fn new(index: &'a ProjectIndex) -> Self {
        Self { index }
    }

    pub fn chunks_for_file(&self, file_path: &str, source: &str) -> Vec<Chunk> {
        let file_idx = match self.locate_file(file_path) {
            Some(idx) => idx,
            None => return Vec::new(),
        };
        // Symbols filtered by file + chunkable kind. Body extraction in
        // `make_chunk` slices `source` via `line_byte_range` (one pass per
        // symbol up to its end_line) — no `lines.collect()` per symbol.
        let approx = self
            .index
            .symbols
            .iter()
            .filter(|s| s.file_idx == file_idx && is_chunkable_kind(s.kind))
            .count();
        let mut out = Vec::with_capacity(approx);
        for sym in self
            .index
            .symbols
            .iter()
            .filter(|s| s.file_idx == file_idx && is_chunkable_kind(s.kind))
        {
            if let Some(c) = self.make_chunk(sym, source) {
                out.push(c);
            }
        }
        out
    }

    fn locate_file(&self, file_path: &str) -> Option<u16> {
        if let Some(i) = self.index.files.iter().position(|f| f.path == file_path) {
            return Some(i as u16);
        }
        let fallback = self
            .index
            .files
            .iter()
            .position(|f| f.path.ends_with(file_path));
        if let Some(i) = fallback {
            tracing::debug!(
                "path fallback: {} matched {}",
                file_path,
                self.index.files[i].path,
            );
        }
        fallback.map(|i| i as u16)
    }

    fn make_chunk(&self, sym: &zti_ts_core::types::Symbol, source: &str) -> Option<Chunk> {
        if sym.line == 0 || sym.end_line < sym.line {
            return None;
        }
        let doc_start = if sym.doc.is_some() {
            find_doc_start_line(source, sym.line)
        } else {
            sym.line
        };

        let range = line_byte_range(source, doc_start, sym.end_line);
        if range.is_empty() {
            return None;
        }
        let raw = &source[range];

        // Single pass over body bytes: emit each line, prepend `<kind>#<id> `
        // to the signature line. No alloc beyond the final body String.
        let mut body = String::with_capacity(raw.len() + 16);
        let sig_offset = (sym.line - doc_start) as usize;
        for (i, line) in raw.split('\n').enumerate() {
            if i > 0 {
                body.push('\n');
            }
            if i == sig_offset {
                let _ = write!(body, "{}#{} ", sym.kind.short(), sym.id);
            }
            body.push_str(line);
        }

        let file = self.index.files.get(sym.file_idx as usize)?;
        let rel_file = file
            .path
            .strip_prefix(&self.index.root)
            .unwrap_or(&file.path)
            .trim_start_matches('/')
            .to_string();

        Some(Chunk {
            file: file.path.clone(),
            rel_file,
            start_line: doc_start,
            end_line: sym.end_line,
            sym_id: sym.id,
            body,
            qualified: sym.qualified.clone(),
            kind: sym.kind,
        })
    }
}

/// Walks back from `sym_line - 1` in `source` over contiguous doc-comment
/// and attribute-style lines, returning the line number where the doc block
/// starts. Regular `//` comments do NOT match, so a file-top license block
/// is naturally excluded. Picks up Rust `///` / `//!` / `/** */` / `*` /
/// `#[…]` and TS/Dart decorators starting with `@`.
fn find_doc_start_line(source: &str, sym_line: u32) -> u32 {
    if sym_line <= 1 {
        return sym_line;
    }
    let range = line_byte_range(source, 1, sym_line - 1);
    let prefix = &source[range];
    let mut back = 0u32;
    for line in prefix.rsplit('\n') {
        let t = line.trim_start();
        if looks_like_doc_or_attr(t) {
            back += 1;
        } else {
            break;
        }
    }
    sym_line - back
}

fn looks_like_doc_or_attr(t: &str) -> bool {
    t.starts_with("///")
        || t.starts_with("//!")
        || t.starts_with("/**")
        || t.starts_with("*/")
        || t.starts_with('*')
        || t.starts_with("#[")
        || t.starts_with('@')
}

/// One whole-file chunk for files we don't parse with tree-sitter (READMEs,
/// docs, plain text). Takes ownership of `content` — no clone.
pub fn chunk_text_file(rel_path: String, full_path: String, content: String) -> Chunk {
    let newlines = content.bytes().filter(|&b| b == b'\n').count() as u32;
    let end_line = if content.is_empty() { 1 } else { newlines + 1 };
    Chunk {
        file: full_path,
        rel_file: rel_path,
        start_line: 1,
        end_line,
        sym_id: Chunk::DOCUMENT_SYM_ID,
        body: content,
        qualified: String::new(),
        kind: Kind::Document,
    }
}

pub fn is_chunkable_kind(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Function
            | Kind::Method
            | Kind::Struct
            | Kind::Enum
            | Kind::TypeAlias
            | Kind::Class
            | Kind::Interface
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rust_source_with_doc() -> &'static str {
        "// Copyright 2024 Foo Corp.\n\
         // Licensed under MIT.\n\
         \n\
         use bar::baz;\n\
         \n\
         /// Encrypts plaintext.\n\
         /// Returns ciphertext.\n\
         #[inline]\n\
         pub fn bytes_encrypt(x: u8) -> u8 {\n\
             x ^ 0xff\n\
         }\n"
    }

    #[test]
    fn find_doc_start_extends_past_docs_and_attrs() {
        // Signature line for `pub fn bytes_encrypt` is line 9.
        let src = rust_source_with_doc();
        let start = find_doc_start_line(src, 9);
        // Walks back over #[inline] (line 8), `///` (lines 6, 7) — three lines.
        assert_eq!(start, 6);
    }

    #[test]
    fn find_doc_start_does_not_swallow_regular_line_comments() {
        // Single-`//` lines at the top of the file (license) must NOT be
        // included in the doc range.
        let src = "// Copyright X\n\
                   // Licensed Y\n\
                   pub fn foo() {}\n";
        // Signature at line 3, no doc above — start should equal sym_line.
        assert_eq!(find_doc_start_line(src, 3), 3);
    }

    #[test]
    fn embed_text_emits_file_header_and_body() {
        let c = Chunk {
            file: "/abs/src/foo.rs".to_string(),
            rel_file: "src/foo.rs".to_string(),
            start_line: 10,
            end_line: 12,
            sym_id: 7,
            body: "f#7 fn bar() {}".to_string(),
            qualified: "foo::bar".to_string(),
            kind: Kind::Function,
        };
        let txt = c.embed_text();
        assert_eq!(&*txt, "FILE: src/foo.rs :10-12\nf#7 fn bar() {}");
    }

    #[test]
    fn chunk_text_file_counts_lines() {
        let c = chunk_text_file(
            "README.md".to_string(),
            "/abs/README.md".to_string(),
            "# Title\n\nSecond para\n".to_string(),
        );
        assert!(c.is_document());
        assert_eq!(c.start_line, 1);
        assert_eq!(c.end_line, 4);
        assert_eq!(c.kind, Kind::Document);
    }

    #[test]
    fn chunk_text_file_empty_is_one_line() {
        let c = chunk_text_file(
            "EMPTY.md".to_string(),
            "/abs/EMPTY.md".to_string(),
            String::new(),
        );
        assert_eq!(c.end_line, 1);
    }

    #[test]
    fn is_chunkable_kind_covers_aggregates() {
        for k in [
            Kind::Function,
            Kind::Method,
            Kind::Struct,
            Kind::Enum,
            Kind::TypeAlias,
            Kind::Class,
            Kind::Interface,
        ] {
            assert!(is_chunkable_kind(k));
        }
        for k in [
            Kind::Module,
            Kind::Field,
            Kind::Variant,
            Kind::Const,
            Kind::Static,
            Kind::Document,
        ] {
            assert!(!is_chunkable_kind(k));
        }
    }
}
