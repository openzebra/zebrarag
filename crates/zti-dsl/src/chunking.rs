use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use zti_common::line_byte_range;
use zti_ts_core::types::Kind;

use crate::model::ProjectIndex;
use crate::render::{build_children_by_parent, render_symbol_rich};

const RICH_MAX_TARGETS: usize = 24;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub sym_id: u32,
    pub header: String,
    pub body: String,
    pub qualified: String,
    pub kind: Kind,
}

impl Chunk {
    pub fn embed_text(&self) -> String {
        format!("{}\n---\n{}\n---\n{}", self.header, self.body, self.header)
    }

    pub fn display_text(&self) -> String {
        format!("{}\n---\n{}", self.header, self.body)
    }
}

pub struct DslChunker<'a> {
    index: &'a ProjectIndex,
    children_by_parent: HashMap<u32, Vec<u32>>,
}

impl<'a> DslChunker<'a> {
    pub fn new(index: &'a ProjectIndex) -> Self {
        Self {
            index,
            children_by_parent: build_children_by_parent(index),
        }
    }

    pub fn chunks_for_file(&self, file_path: &str, source: &str) -> Vec<Chunk> {
        let file_idx = match self.locate_file(file_path) {
            Some(idx) => idx,
            None => return Vec::new(),
        };
        // Each chunk slice goes through `line_byte_range`, which scans the
        // source up to the requested end line once. No `lines.collect()` + per-
        // symbol `join("\n")`; the cost is paid lazily and only for symbols we
        // actually keep.
        self.index
            .symbols
            .iter()
            .filter(|s| s.file_idx == file_idx && is_chunkable_kind(s.kind))
            .filter_map(|s| self.make_chunk(s, source))
            .collect()
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
        let range = line_byte_range(source, sym.line, sym.end_line);
        if range.is_empty() {
            return None;
        }
        let body = source[range].to_string();

        let mut header = String::with_capacity(384);
        render_symbol_rich(
            self.index,
            sym.id,
            &self.children_by_parent,
            RICH_MAX_TARGETS,
            &mut header,
        );

        let file = self.index.files.get(sym.file_idx as usize)?;
        Some(Chunk {
            file: file.path.clone(),
            start_line: sym.line,
            end_line: sym.end_line,
            sym_id: sym.id,
            header,
            body,
            qualified: sym.qualified.clone(),
            kind: sym.kind,
        })
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

    fn synthetic_chunk() -> Chunk {
        Chunk {
            file: "src/foo.rs".to_string(),
            start_line: 10,
            end_line: 12,
            sym_id: 7,
            header: "f#7 foo::bar".to_string(),
            body: "fn bar() {}\n// body".to_string(),
            qualified: "foo::bar".to_string(),
            kind: Kind::Function,
        }
    }

    #[test]
    fn embed_text_brackets_body_with_header() {
        let c = synthetic_chunk();
        let txt = c.embed_text();
        assert!(txt.starts_with("f#7 foo::bar\n---\n"));
        assert!(txt.ends_with("\n---\nf#7 foo::bar"));
        assert!(txt.contains("fn bar()"));
    }

    #[test]
    fn display_text_one_header_then_body() {
        let c = synthetic_chunk();
        let txt = c.display_text();
        assert_eq!(txt, "f#7 foo::bar\n---\nfn bar() {}\n// body");
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
        for k in [Kind::Module, Kind::Field, Kind::Variant, Kind::Const, Kind::Static] {
            assert!(!is_chunkable_kind(k));
        }
    }
}
