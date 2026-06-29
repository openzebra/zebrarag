mod atom;
mod merge;
mod positions;

pub struct ChunkConfig {
    pub chunk_size: usize,
    pub min_chunk_size: usize,
    pub chunk_overlap: usize,
}

pub struct SubChunk {
    pub byte_start: usize,
    pub byte_end: usize,
    pub start_line: u32,
    pub end_line: u32,
}

use crate::merge::{chunk_text, chunk_text_with_ts};

pub fn split_text(
    source: &str,
    config: &ChunkConfig,
    lang: Option<&tree_sitter::Language>,
    terminal_kinds: &[u16],
) -> Vec<SubChunk> {
    let min_chunk = config.min_chunk_size;
    let overlap = config.chunk_overlap.min(min_chunk);

    if let Some(ts_lang) = lang {
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(ts_lang).is_err() {
            // fall through to regex
        } else if let Some(tree) = parser.parse(source, None) {
            return chunk_text_with_ts(
                source,
                config.chunk_size,
                overlap,
                min_chunk,
                tree.root_node(),
                terminal_kinds,
            );
        }
    }

    chunk_text(source, config.chunk_size, overlap, min_chunk)
}
