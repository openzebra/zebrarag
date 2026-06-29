pub mod batch;
pub mod chunking;
pub mod index;
pub mod model;
pub mod render;
pub mod search_dep;

pub use batch::resolve_symbol_bodies;
pub use chunking::{Chunk, ChunkStrategy, DslChunker};
pub use index::{
    SourceFile, build_index, build_index_from_sources, files_by_language, filter_files,
    glob_match_files,
};
pub use model::{FileEntry, ProjectIndex};
pub use render::tree::AsciiTreeRenderer;
pub use search_dep::{NameMatch, render_symbol_overview, resolve_name};
pub use zrag_tree_sitter::{Language, detect_from_path};
pub use zrag_ts_core::types::{Edge, EdgeKind, Kind, Symbol, Target};
