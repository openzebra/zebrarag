pub mod chunking;
pub mod index;
pub mod model;
pub mod render;

pub use chunking::{Chunk, DslChunker};
pub use index::build_index;
pub use model::{FileEntry, ProjectIndex};
pub use render::tree::AsciiTreeRenderer;
pub use render::{render_symbol_inline, InlineOpts, LEGEND_LINE};
pub use zti_ts_core::types::{Edge, EdgeKind, Kind, Symbol, Target};
pub use zti_tree_sitter::{Language, detect_from_path};
