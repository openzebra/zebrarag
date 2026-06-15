use std::borrow::Cow;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use zti_common::LineIndex;
pub use zti_common::chunk_strategy::ChunkStrategy;
use zti_ts_core::types::Kind;

use crate::model::ProjectIndex;
use crate::render::dsl::load_manifest_content;

pub fn find_manifest(root: &Path) -> Option<String> {
    crate::index::MANIFEST_NAMES.iter().find_map(|name| {
        let p = root.join(name);
        std::fs::read_to_string(&p).ok()
    })
}

pub fn write_preamble(index: &ProjectIndex, out: &mut String) {
    for rel in &index.manifest_paths {
        if let Some(content) = load_manifest_content(&index.root, rel) {
            let _ = writeln!(out, "@ {}\n{}", rel, content);
            out.push('\n');
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk<'a> {
    pub file: Arc<str>,
    pub rel_file: Arc<str>,
    pub start_line: u32,
    pub end_line: u32,
    pub sym_id: u32,
    pub sub_chunk_idx: u32,
    pub total_sub_chunks: u32,
    pub chunk_strategy: ChunkStrategy,
    pub body: Cow<'a, str>,
    pub qualified: String,
    pub kind: Kind,
}

pub struct DslChunker<'a> {
    index: &'a ProjectIndex,
    symbols_by_file: FxHashMap<u16, Vec<&'a zti_ts_core::types::Symbol>>,
}

impl<'a> DslChunker<'a> {
    pub fn new(index: &'a ProjectIndex) -> Self {
        let mut symbols_by_file: FxHashMap<u16, Vec<&'a zti_ts_core::types::Symbol>> =
            FxHashMap::with_capacity_and_hasher(index.files.len(), rustc_hash::FxBuildHasher);
        for sym in &index.symbols {
            symbols_by_file.entry(sym.file_idx).or_default().push(sym);
        }
        Self {
            index,
            symbols_by_file,
        }
    }

    pub fn chunks_for_file<'s>(&self, file_path: &str, source: &'s str) -> Vec<Chunk<'s>> {
        let file_idx = match self.locate_file(file_path) {
            Some(idx) => idx,
            None => return Vec::new(),
        };
        let Some(symbols) = self.symbols_by_file.get(&file_idx) else {
            return Vec::new();
        };
        let file_info = &self.index.files[file_idx as usize];
        let file_arc: Arc<str> = Arc::from(file_info.path.as_str());
        let rel = file_info
            .path
            .strip_prefix(&self.index.root)
            .unwrap_or(&file_info.path)
            .trim_start_matches('/');
        let rel_file_arc: Arc<str> = Arc::from(rel);

        let approx = symbols.iter().filter(|s| is_chunkable_kind(s.kind)).count();
        let mut out = Vec::with_capacity(approx);
        let line_index = LineIndex::new(source);
        for sym in symbols.iter().filter(|s| is_chunkable_kind(s.kind)) {
            if let Some(c) = self.make_chunk(sym, source, &line_index, &file_arc, &rel_file_arc) {
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

    fn make_chunk<'s>(
        &self,
        sym: &zti_ts_core::types::Symbol,
        source: &'s str,
        line_index: &LineIndex,
        file: &Arc<str>,
        rel_file: &Arc<str>,
    ) -> Option<Chunk<'s>> {
        if sym.line == 0 || sym.end_line < sym.line {
            return None;
        }
        let doc_start = if sym.doc.is_some() {
            find_doc_start_line(source, sym.line, line_index)
        } else {
            sym.line
        };

        let range = line_index.byte_range(doc_start, sym.end_line);
        if range.is_empty() {
            return None;
        }
        let body = Cow::Borrowed(&source[range]);

        Some(Chunk {
            file: Arc::clone(file),
            rel_file: Arc::clone(rel_file),
            start_line: doc_start,
            end_line: sym.end_line,
            sym_id: sym.id,
            sub_chunk_idx: 0,
            total_sub_chunks: 1,
            chunk_strategy: ChunkStrategy::Symbol,
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
pub(crate) fn find_doc_start_line(source: &str, sym_line: u32, line_index: &LineIndex) -> u32 {
    if sym_line <= 1 {
        return sym_line;
    }
    let range = line_index.byte_range(1, sym_line - 1);
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
/// docs, plain text). Borrows `content` — no clone.
pub fn chunk_text_file<'a>(
    rel_path: &str,
    full_path: &str,
    content: &'a str,
) -> Chunk<'a> {
    let newlines = content.bytes().filter(|&b| b == b'\n').count() as u32;
    let end_line = if content.is_empty() { 1 } else { newlines + 1 };
    Chunk {
        file: Arc::from(full_path),
        rel_file: Arc::from(rel_path),
        start_line: 1,
        end_line,
        sym_id: u32::MAX,
        sub_chunk_idx: 0,
        total_sub_chunks: 1,
        chunk_strategy: ChunkStrategy::Symbol,
        body: Cow::Borrowed(content),
        qualified: String::new(),
        kind: Kind::Document,
    }
}

/// One accumulated run of consecutive TSV rows plus the physical line span it
/// covers. Finalized into a single `Chunk` once it reaches the byte budget.
struct RowGroup {
    buf: String,
    first: u32,
    last: u32,
}

/// State threaded through the packing fold: finished chunks plus the group
/// currently being filled.
struct PackState {
    done: Vec<Chunk<'static>>,
    pending: Option<RowGroup>,
}

/// Finalize an accumulated row group into a `Chunk`, moving its buffer into
/// `Cow::Owned` (no copy). The body is the raw row text only — no column labels.
fn finalize_group(rel_path: &Arc<str>, full_path: &Arc<str>, group: RowGroup) -> Chunk<'static> {
    Chunk {
        file: Arc::clone(full_path),
        rel_file: Arc::clone(rel_path),
        start_line: group.first,
        end_line: group.last,
        sym_id: u32::MAX,
        sub_chunk_idx: 0,
        total_sub_chunks: 1,
        chunk_strategy: ChunkStrategy::Symbol,
        qualified: format!("{rel_path}:rows:{}-{}", group.first, group.last),
        body: Cow::Owned(group.buf),
        kind: Kind::Document,
    }
}

/// Row-aware chunks for a tabular file (TSV/PSV). The first physical line is
/// the header and is dropped; each subsequent non-empty line is a record
/// embedded as its raw text. Consecutive records are greedily packed into one
/// chunk until adding the next would exceed `target_bytes` (never split
/// mid-record). The delimiter is irrelevant here — packing is line-based — so
/// this serves both tab- and pipe-separated inputs. A single row larger than
/// `target_bytes` becomes its own chunk and is split downstream.
pub fn chunk_tabular_file(
    rel_path: &str,
    full_path: &str,
    content: &str,
    target_bytes: usize,
) -> Vec<Chunk<'static>> {
    let rel_arc: Arc<str> = Arc::from(rel_path);
    let full_arc: Arc<str> = Arc::from(full_path);
    let cap = content.len() / target_bytes.max(1) + 1;
    // `enumerate` before `skip(1)` keeps `i` as the 0-based physical line, so the
    // first data row (physical line 2) lands at `i == 1`.
    let state = content
        .lines()
        .enumerate()
        .skip(1)
        .filter(|(_, line)| !line.is_empty())
        .fold(
            PackState {
                done: Vec::with_capacity(cap),
                pending: None,
            },
            |mut state, (i, line)| {
                let phys = i as u32 + 1;
                let start_new = match state.pending.as_ref() {
                    Some(group) => group.buf.len() + 1 + line.len() > target_bytes,
                    None => true,
                };
                if start_new {
                    if let Some(group) = state.pending.take() {
                        state.done.push(finalize_group(&rel_arc, &full_arc, group));
                    }
                    let mut buf = String::with_capacity(target_bytes.max(line.len()));
                    buf.push_str(line);
                    state.pending = Some(RowGroup {
                        buf,
                        first: phys,
                        last: phys,
                    });
                } else if let Some(group) = state.pending.as_mut() {
                    group.buf.push('\n');
                    group.buf.push_str(line);
                    group.last = phys;
                }
                state
            },
        );

    let PackState { mut done, pending } = state;
    if let Some(group) = pending {
        done.push(finalize_group(&rel_arc, &full_arc, group));
    }
    done
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
        let src = rust_source_with_doc();
        let idx = LineIndex::new(src);
        let start = find_doc_start_line(src, 9, &idx);
        assert_eq!(start, 6);
    }

    #[test]
    fn find_doc_start_does_not_swallow_regular_line_comments() {
        let src = "// Copyright X\n\
                   // Licensed Y\n\
                   pub fn foo() {}\n";
        let idx = LineIndex::new(src);
        assert_eq!(find_doc_start_line(src, 3, &idx), 3);
    }

    #[test]
    fn chunk_text_file_counts_lines() {
        let c = chunk_text_file("README.md", "/abs/README.md", "# Title\n\nSecond para\n");
        assert_eq!(c.kind, Kind::Document);
        assert_eq!(c.start_line, 1);
        assert_eq!(c.end_line, 4);
    }

    #[test]
    fn chunk_text_file_empty_is_one_line() {
        let c = chunk_text_file("EMPTY.md", "/abs/EMPTY.md", "");
        assert_eq!(c.end_line, 1);
    }

    #[test]
    fn chunk_tabular_packs_small_rows_into_one_chunk() {
        let content = "id\tname\tnote\n1\talice\thi\n2\tbob\tyo\n";
        // Generous budget → both data rows pack into a single chunk.
        let chunks = chunk_tabular_file("db/users.tsv", "/abs/db/users.tsv", content, 4096);
        assert_eq!(chunks.len(), 1);

        let c = &chunks[0];
        assert_eq!(c.kind, Kind::Document);
        assert_eq!(&*c.rel_file, "db/users.tsv");
        // Spans physical lines 2..=3 (header is line 1).
        assert_eq!(c.start_line, 2);
        assert_eq!(c.end_line, 3);
        assert_eq!(c.qualified, "db/users.tsv:rows:2-3");
        // Raw values only, records joined by newline — no column labels.
        assert_eq!(c.body.as_ref(), "1\talice\thi\n2\tbob\tyo");
    }

    #[test]
    fn chunk_tabular_starts_new_chunk_at_budget() {
        let content = "id\tname\n1\talice\n2\tbob\n3\tcarol\n";
        // Budget holds one ~7-byte record but not two → one chunk per record.
        let chunks = chunk_tabular_file("u.tsv", "/abs/u.tsv", content, 8);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].body.as_ref(), "1\talice");
        assert_eq!(chunks[0].start_line, 2);
        assert_eq!(chunks[0].end_line, 2);
        assert_eq!(chunks[0].qualified, "u.tsv:rows:2-2");
        assert_eq!(chunks[2].body.as_ref(), "3\tcarol");
        assert_eq!(chunks[2].start_line, 4);
    }

    #[test]
    fn chunk_tabular_oversized_row_stands_alone() {
        // A record larger than the budget is emitted on its own (the indexer
        // splits it downstream); neighbours do not merge into it.
        let content = "h\nshort\nthisrowislongerthanbudget\ntiny\n";
        let chunks = chunk_tabular_file("o.tsv", "/abs/o.tsv", content, 10);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].body.as_ref(), "thisrowislongerthanbudget");
        assert_eq!(chunks[1].start_line, 3);
        assert_eq!(chunks[1].end_line, 3);
    }

    #[test]
    fn chunk_tabular_header_only_or_empty_yields_nothing() {
        assert!(chunk_tabular_file("h.tsv", "/abs/h.tsv", "a\tb\tc\n", 4096).is_empty());
        assert!(chunk_tabular_file("e.tsv", "/abs/e.tsv", "", 4096).is_empty());
    }

    #[test]
    fn chunk_tabular_skips_blank_lines_keeping_physical_line_numbers() {
        // Blank line 3 is skipped; the second record keeps its physical line 4.
        let content = "a\tb\n1\t2\n\n3\t4\n";
        // Tiny budget → one chunk per record so each line span is assertable.
        let chunks = chunk_tabular_file("r.tsv", "/abs/r.tsv", content, 3);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_line, 2);
        assert_eq!(chunks[0].body.as_ref(), "1\t2");
        assert_eq!(chunks[1].start_line, 4);
        assert_eq!(chunks[1].body.as_ref(), "3\t4");
    }

    #[test]
    fn chunk_tabular_packs_pipe_rows() {
        let content = "id|name|note\n1|alice|hi\n2|bob|yo\n";
        let chunks = chunk_tabular_file("db/users.psv", "/abs/db/users.psv", content, 4096);
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.start_line, 2);
        assert_eq!(c.end_line, 3);
        assert_eq!(c.qualified, "db/users.psv:rows:2-3");
        assert_eq!(c.body.as_ref(), "1|alice|hi\n2|bob|yo");
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
