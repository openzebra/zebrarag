pub mod dsl;
pub mod tree;

pub const CHARS_PER_TOKEN: usize = 4;

pub use dsl::{
    build_children_by_parent, format_target, render_symbol_inline, render_symbol_rich, InlineOpts,
    LEGEND_LINE,
};
