pub mod chunking;
pub mod index;
pub mod model;
pub mod render;

pub use chunking::{Chunk, DslChunker};
pub use index::build_index;
pub use model::{Edge, EdgeKind, FileEntry, Kind, Language, ProjectIndex, Symbol, Target};
pub use render::tree::AsciiTreeRenderer;
pub use render::{render_symbol_inline, InlineOpts, LEGEND_LINE};
