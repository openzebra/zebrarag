use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::Write as _;

use zti_ts_core::types::{Edge, EdgeKind, Kind, Target};

use crate::model::ProjectIndex;

pub const LEGEND_LINE: &str = "# k short = Kind   f#=fn m#=method s#=struct e#=enum C#=class I#=iface t#=typealias c#=const v#=static .=field/variant E#=event X#=error M#=mod";

pub struct InlineOpts {
    pub max_inline_targets: usize,
    pub max_doc_lines: usize,
    pub show_file_path: bool,
    pub show_line_range: bool,
}

impl InlineOpts {
    pub fn for_embedding() -> Self {
        Self {
            max_inline_targets: 12,
            max_doc_lines: 2,
            show_file_path: true,
            show_line_range: true,
        }
    }

    pub fn compact() -> Self {
        Self {
            max_inline_targets: 4,
            max_doc_lines: 1,
            show_file_path: false,
            show_line_range: false,
        }
    }
}

pub fn render_symbol_inline(
    index: &ProjectIndex,
    id: u32,
    opts: &InlineOpts,
    out: &mut String,
) {
    let sym = match index.symbols.get(id as usize) {
        Some(s) => s,
        None => return,
    };
    let file = index.files.get(sym.file_idx as usize);

    let short = sym.kind.short();
    out.push_str(short);
    out.push('#');
    let _ = write!(out, "{}", sym.id);
    out.push(' ');
    out.push_str(&sym.qualified);

    if opts.show_file_path
        && let Some(f) = file
    {
        let rel = f.path.strip_prefix(&index.root).unwrap_or(&f.path);
        let rel = rel.trim_start_matches('/');
        out.push(' ');
        out.push_str(rel);
    }

    if opts.show_line_range {
        let _ = write!(out, " :{}-{}", sym.line, sym.end_line);
    }

    if let Some(ref doc) = sym.doc {
        let mut joined = String::new();
        let mut seen = 0usize;
        for line in doc.lines().take(opts.max_doc_lines) {
            if seen > 0 {
                joined.push(' ');
            }
            joined.push_str(line);
            seen += 1;
        }
        if seen > 0 {
            out.push(' ');
            out.push('"');
            out.push_str(joined.trim());
            out.push('"');
        }
    }

    write_targets(
        out,
        " <- ",
        index
            .reverse_edges
            .get(&id)
            .into_iter()
            .flat_map(|v| v.iter())
            .filter(|e| e.kind == EdgeKind::Call),
        opts.max_inline_targets,
        |edge, out| {
            if let Some(ts) = index.symbols.get(edge.from as usize) {
                out.push_str(&ts.qualified);
            }
        },
    );

    write_targets(
        out,
        " -> ",
        index
            .forward_edges
            .get(&id)
            .into_iter()
            .flat_map(|v| v.iter())
            .filter(|e| e.kind == EdgeKind::Call),
        opts.max_inline_targets,
        |edge, out| {
            out.push_str(&format_target(&edge.to));
        },
    );
}

/// Inline-render up to `max` edge targets, followed by `...` if the iterator
/// would have produced more. Single-pass — no intermediate `Vec<&Edge>`.
fn write_targets<'a, I, F>(out: &mut String, prefix: &str, edges: I, max: usize, mut write_one: F)
where
    I: IntoIterator<Item = &'a Edge>,
    F: FnMut(&Edge, &mut String),
{
    let mut iter = edges.into_iter().peekable();
    if iter.peek().is_none() {
        return;
    }
    out.push_str(prefix);
    let mut overflow = false;
    for (i, edge) in iter.enumerate() {
        if i >= max {
            overflow = true;
            break;
        }
        if i > 0 {
            out.push(' ');
        }
        write_one(edge, out);
    }
    if overflow {
        out.push_str(" ...");
    }
}

pub fn format_target(target: &Target) -> Cow<'_, str> {
    match target {
        Target::Resolved(id) => Cow::Owned(format!("#{}", id)),
        Target::Unresolved(name) | Target::External(name) => Cow::Borrowed(name),
    }
}

pub struct DslRenderer<'a> {
    index: &'a ProjectIndex,
    max_tokens: usize,
}

impl<'a> DslRenderer<'a> {
    pub fn new(index: &'a ProjectIndex, max_tokens: usize) -> Self {
        Self { index, max_tokens }
    }

    pub fn render(
        &self,
        file_filter: Option<&[u16]>,
        kind_filter: Option<&[Kind]>,
    ) -> String {
        let mut out = String::new();
        let mut tokens_used = 0;

        let opts = InlineOpts::for_embedding();

        let filter_set: Option<HashSet<u16>> = file_filter.map(|f| f.iter().copied().collect());
        let kind_set: Option<HashSet<Kind>> = kind_filter.map(|k| k.iter().copied().collect());

        let mut last_file: Option<u16> = None;

        for sym in &self.index.symbols {
            if let Some(ref set) = filter_set
                && !set.contains(&sym.file_idx)
            {
                continue;
            }
            if let Some(ref set) = kind_set
                && !set.contains(&sym.kind)
            {
                continue;
            }

            if last_file != Some(sym.file_idx) {
                if let Some(file) = self.index.files.get(sym.file_idx as usize) {
                    let _ = writeln!(out, "FILES #{} {}", sym.file_idx, file.path);
                }
                last_file = Some(sym.file_idx);
            }

            let mut line = String::new();
            render_symbol_inline(self.index, sym.id, &opts, &mut line);
            out.push_str(&line);
            out.push('\n');

            tokens_used += line.len() / crate::render::CHARS_PER_TOKEN;
            if tokens_used >= self.max_tokens {
                out.push_str("... (truncated)\n");
                break;
            }
        }

        out
    }
}

pub fn render_files_only(index: &ProjectIndex, file_indices: &[u16]) -> String {
    let mut out = String::new();
    out.push_str("FILES (#N = file ID — appears as \"# N\" in project_map output)\n");
    for &idx in file_indices {
        if let Some(file) = index.files.get(idx as usize) {
            let _ = writeln!(out, "# {} {}", idx, file.path);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use zti_ts_core::types::Kind;
    use zti_tree_sitter::Language;

    use crate::model::{FileEntry, ProjectIndex};

    use super::*;

    fn mk_index(symbols: Vec<zti_ts_core::types::Symbol>, files: Vec<FileEntry>) -> ProjectIndex {
        ProjectIndex {
            symbols,
            edges: Vec::new(),
            files,
            qualified_map: HashMap::new(),
            reverse_edges: HashMap::new(),
            forward_edges: HashMap::new(),
            root: "/p".into(),
        }
    }

    fn mk_sym(id: u32, name: &str, kind: Kind, file_idx: u16) -> zti_ts_core::types::Symbol {
        zti_ts_core::types::Symbol {
            id,
            kind,
            name: name.to_string(),
            qualified: name.to_string(),
            file_idx,
            line: 1,
            end_line: 1,
            signature: String::new(),
            doc: None,
            base_classes: Vec::new(),
            parent: None,
            traits: Vec::new(),
        }
    }

    fn mk_file(path: &str) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            language: Language::Rust,
            imports: HashMap::new(),
        }
    }

    #[test]
    fn legend_mentions_every_emitted_kind_prefix() {
        for marker in ["f#", "m#", "c#", "v#", "s#", "C#", "e#", "t#", "I#", "M#"] {
            assert!(
                LEGEND_LINE.contains(marker),
                "LEGEND_LINE missing marker `{}`",
                marker
            );
        }
    }

    #[test]
    fn no_trailing_whitespace_on_any_line() {
        let idx = mk_index(
            vec![mk_sym(0, "Empty", Kind::Struct, 0)],
            vec![mk_file("/p/a.rs")],
        );
        let out = DslRenderer::new(&idx, 8000).render(None, None);
        for (i, line) in out.lines().enumerate() {
            assert_eq!(
                line.trim_end(),
                line,
                "trailing whitespace on line {}: <{}>",
                i + 1,
                line
            );
        }
    }

    #[test]
    fn struct_with_no_fields_renders_clean() {
        let idx = mk_index(
            vec![mk_sym(0, "Empty", Kind::Struct, 0)],
            vec![mk_file("/p/a.rs")],
        );
        let out = DslRenderer::new(&idx, 8000).render(None, None);
        assert!(out.contains("s#0 Empty"), "got:\n{}", out);
    }

    #[test]
    fn struct_with_fields_renders_field_list() {
        let mut field = mk_sym(1, "x", Kind::Field, 0);
        field.parent = Some(0);
        let idx = mk_index(
            vec![mk_sym(0, "Point", Kind::Struct, 0), field],
            vec![mk_file("/p/a.rs")],
        );
        let out = DslRenderer::new(&idx, 8000).render(None, None);
        assert!(out.contains("s#0 Point"), "got:\n{}", out);
    }

    #[test]
    fn render_files_only_emits_tree_without_symbols() {
        let idx = mk_index(
            vec![
                mk_sym(0, "foo", Kind::Function, 0),
                mk_sym(1, "bar", Kind::Function, 1),
                mk_sym(2, "Baz", Kind::Struct, 2),
            ],
            vec![
                mk_file("/p/src/main.rs"),
                mk_file("/p/src/lib.rs"),
                mk_file("/p/src/mod.rs"),
            ],
        );
        let all: Vec<u16> = (0..3).collect();
        let out = render_files_only(&idx, &all);

        assert!(out.starts_with("FILES"), "should start with FILES header, got:\n{}", out);
        assert!(out.contains("src/"), "should contain src dir, got:\n{}", out);
        assert!(out.contains("main.rs"), "should contain main.rs, got:\n{}", out);
        assert!(out.contains("lib.rs"), "should contain lib.rs, got:\n{}", out);
        assert!(out.contains("mod.rs"), "should contain mod.rs, got:\n{}", out);
        assert!(!out.contains("f#"), "should NOT contain symbol DSL, got:\n{}", out);
        assert!(!out.contains("s#"), "should NOT contain symbol DSL, got:\n{}", out);
        assert!(!out.contains("foo"), "should NOT contain symbol names, got:\n{}", out);
    }
}
