use serde::{Deserialize, Serialize};

use crate::model::{Kind, ProjectIndex};
use crate::render::{render_symbol_inline, InlineOpts};

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
    opts: InlineOpts,
}

impl<'a> DslChunker<'a> {
    pub fn new(index: &'a ProjectIndex) -> Self {
        Self {
            index,
            opts: InlineOpts::for_embedding(),
        }
    }

    pub fn chunks_for_file(&self, file_path: &str, source: &str) -> Vec<Chunk> {
        let file_idx = match self.locate_file(file_path) {
            Some(idx) => idx,
            None => return Vec::new(),
        };
        let lines: Vec<&str> = source.lines().collect();
        self.index
            .symbols
            .iter()
            .filter(|s| s.file_idx == file_idx && is_chunkable_kind(s.kind))
            .filter_map(|s| self.make_chunk(s, &lines))
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

    fn make_chunk(&self, sym: &crate::model::Symbol, lines: &[&str]) -> Option<Chunk> {
        let start = (sym.line as usize).saturating_sub(1);
        let end = (sym.end_line as usize).min(lines.len());
        if start >= end {
            return None;
        }
        let body_src = lines[start..end].join("\n");
        let mut header = String::with_capacity(256);
        render_symbol_inline(self.index, sym.id, &self.opts, &mut header);
        let file = self.index.files.get(sym.file_idx as usize)?;
        Some(Chunk {
            file: file.path.clone(),
            start_line: sym.line,
            end_line: sym.end_line,
            sym_id: sym.id,
            header,
            body: body_src,
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
