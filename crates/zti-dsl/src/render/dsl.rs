use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use zti_tree_sitter::Language;
use zti_ts_core::types::{EdgeKind, Kind, Target};

use crate::model::ProjectIndex;
use crate::render::MANIFEST_CAP;

pub const LEGEND_LINE: &str = "# k short = Kind   f#=fn m#=method s#=struct e#=enum C#=class I#=iface t#=typealias c#=const v#=static .=field/variant E#=event X#=error M#=mod   PKG = project manifest";

pub const AST_HEADER: &str = "\
Code AST. Use node #ID to fetch full code body.\n";

const RUST_LEGEND: &str = "\
Types: @=File, f=fn, m=method, s=struct, c=class/impl, t=trait, e=enum, v=var/field/variant, d=mod, k=const, y=typealias\n";

/// O(N) once-per-index map: parent_id -> child ids. Used by render_symbol_rich
/// so the siblings line is O(siblings) not O(all symbols). Children are sorted
/// by line at build time so render_node and rich-header siblings consume them
/// in source order without per-call allocations.
pub fn build_children_by_parent(index: &ProjectIndex) -> HashMap<u32, Vec<u32>> {
    let mut map: HashMap<u32, Vec<u32>> = HashMap::with_capacity(index.symbols.len() / 4);
    for sym in &index.symbols {
        if let Some(p) = sym.parent {
            map.entry(p).or_default().push(sym.id);
        }
    }
    for ids in map.values_mut() {
        ids.sort_by_key(|cid| {
            index
                .symbols
                .get(*cid as usize)
                .map(|s| s.line)
                .unwrap_or(0)
        });
    }
    map
}

fn rust_short(kind: Kind) -> &'static str {
    match kind {
        Kind::Function => "f",
        Kind::Method => "m",
        Kind::Struct => "s",
        Kind::Enum => "e",
        Kind::Interface => "t",
        Kind::Module => "d",
        Kind::Static | Kind::Field | Kind::Variant => "v",
        Kind::Const => "k",
        Kind::TypeAlias => "y",
        Kind::Class | Kind::Impl => "c",
        Kind::Event => "E",
        Kind::Error => "X",
        Kind::Document => "@",
    }
}

fn short_for(kind: Kind, lang: Language) -> &'static str {
    match lang {
        Language::Rust => rust_short(kind),
        _ => kind.short(),
    }
}

pub(crate) fn lang_label(lang: Language) -> &'static str {
    match lang {
        Language::Rust => "Rust",
        Language::Ts => "TypeScript",
        Language::Tsx => "TypeScript",
        Language::Dart => "Dart",
        Language::Solidity => "Solidity",
    }
}

const GENERIC_LEGEND: &str = "\
Types: @=File, C#=class, I#=interface, s#=struct, e#=enum, f#=fn, m#=method, v#=var/field/variant, c#=const, d#=mod, t#=typealias, E#=event, X#=error\n";

pub(crate) fn lang_legend(lang: Language) -> &'static str {
    match lang {
        Language::Rust => RUST_LEGEND,
        _ => GENERIC_LEGEND,
    }
}

fn symbol_or_descendant_matches(
    index: &ProjectIndex,
    children: &HashMap<u32, Vec<u32>>,
    id: u32,
    kind_set: &HashSet<Kind>,
) -> bool {
    let Some(sym) = index.symbols.get(id as usize) else {
        return false;
    };
    if kind_set.contains(&sym.kind) {
        return true;
    }
    if let Some(child_ids) = children.get(&id) {
        for &cid in child_ids {
            if symbol_or_descendant_matches(index, children, cid, kind_set) {
                return true;
            }
        }
    }
    false
}

pub(crate) fn load_manifest_content(root: &str, rel_path: &str) -> Option<String> {
    let full = if rel_path.starts_with('/') {
        let mut s = String::with_capacity(rel_path.len());
        s.push_str(rel_path);
        s
    } else {
        let root_trim = root.trim_end_matches('/');
        let mut s = String::with_capacity(root_trim.len() + 1 + rel_path.len());
        s.push_str(root_trim);
        s.push('/');
        s.push_str(rel_path);
        s
    };
    let content = std::fs::read_to_string(&full).ok()?;
    if content.len() > MANIFEST_CAP {
        let end = content.ceil_char_boundary(MANIFEST_CAP);
        let mut capped = String::with_capacity(end + 4);
        capped.push_str(&content[..end]);
        capped.push_str("\n...");
        Some(capped)
    } else {
        Some(content)
    }
}

const SIG_TAIL_TRIM: &[char] = &['{', ';', ',', ' ', '\t'];

// Order matters: longer-overlapping prefixes come first because the inner
// match-and-break consumes the first hit per pass. The outer loop repeats
// until no prefix matches, so chained keywords ("pub async fn …",
// "export async function …") collapse to the bare signature.
static RUST_PREFIXES: &[&str] = &[
    "pub(crate) ",
    "pub(super) ",
    "pub ",
    "unsafe fn ",
    "async fn ",
    "const fn ",
    "fn ",
    "unsafe ",
    "async ",
    "const ",
    "static ",
    "let ",
    "struct ",
    "enum ",
    "trait ",
    "mod ",
    "impl ",
    "type ",
];

static TS_PREFIXES: &[&str] = &[
    "export default ",
    "export ",
    "declare ",
    "public ",
    "private ",
    "protected ",
    "async function ",
    "function ",
    "async ",
    "static ",
    "abstract ",
    "readonly ",
    "class ",
    "interface ",
    "enum ",
    "type ",
    "const ",
    "let ",
    "var ",
];

// Deliberately omits return types ("void ", "String ", "Future<…> ", …):
// stripping them destroys reader-useful info, and enumerating every generic
// container is hopeless.
static DART_PREFIXES: &[&str] = &[
    "abstract ",
    "external ",
    "factory ",
    "static ",
    "late ",
    "final ",
    "const ",
    "class ",
    "mixin ",
    "enum ",
    "extension ",
    "typedef ",
];

// Deliberately omits primitive types ("uint256 ", "address ", …): for state
// variables they are part of the signature, not boilerplate.
static SOLIDITY_PREFIXES: &[&str] = &[
    "public ",
    "private ",
    "internal ",
    "external ",
    "pure ",
    "view ",
    "payable ",
    "virtual ",
    "override ",
    "function ",
    "modifier ",
    "event ",
    "error ",
    "struct ",
    "contract ",
    "interface ",
    "library ",
];

fn strip_prefixes_loop<'a>(mut s: &'a str, prefixes: &[&str]) -> &'a str {
    loop {
        let before = s.as_ptr();
        for &p in prefixes {
            if let Some(rest) = s.strip_prefix(p) {
                s = rest;
                break;
            }
        }
        if s.as_ptr() == before {
            return s;
        }
    }
}

fn format_signature(sig: &str, lang: Language) -> &str {
    let trimmed = sig.trim();
    if trimmed.is_empty() {
        return trimmed;
    }
    let prefixes = match lang {
        Language::Rust => RUST_PREFIXES,
        Language::Ts | Language::Tsx => TS_PREFIXES,
        Language::Dart => DART_PREFIXES,
        Language::Solidity => SOLIDITY_PREFIXES,
    };
    strip_prefixes_loop(trimmed, prefixes).trim_end_matches(SIG_TAIL_TRIM)
}

fn render_node(
    index: &ProjectIndex,
    children: &HashMap<u32, Vec<u32>>,
    id: u32,
    depth: usize,
    out: &mut String,
    lang: Language,
    max_bytes: usize,
) -> bool {
    let Some(sym) = index.symbols.get(id as usize) else {
        return false;
    };

    for _ in 0..depth {
        out.push_str("  ");
    }

    out.push_str(short_for(sym.kind, lang));
    out.push('#');
    let _ = write!(out, "{}", sym.id);
    out.push(' ');

    if sym.kind == Kind::Impl {
        out.push_str(&sym.name);
    } else {
        let sig = format_signature(&sym.signature, lang);
        if sig.is_empty() {
            out.push_str(&sym.name);
        } else {
            out.push_str(sig);
        }
    }

    if sym.kind == Kind::Module && sym.line == sym.end_line {
        let raw = sym.signature.trim();
        if raw.ends_with(';') && !raw.contains('{') {
            out.push(';');
        }
    }

    let _ = write!(out, " :{}-{}", sym.line, sym.end_line);
    out.push('\n');

    if out.len() >= max_bytes {
        return true;
    }

    if let Some(child_ids) = children.get(&id) {
        for &cid in child_ids {
            if render_node(index, children, cid, depth + 1, out, lang, max_bytes) {
                return true;
            }
        }
    }
    false
}

pub fn render_symbol_rich(
    index: &ProjectIndex,
    id: u32,
    children_by_parent: &HashMap<u32, Vec<u32>>,
    max_targets: usize,
    out: &mut String,
) {
    let Some(sym) = index.symbols.get(id as usize) else {
        return;
    };
    let file = index.files.get(sym.file_idx as usize);

    out.push_str(sym.kind.short());
    out.push('#');
    let _ = write!(out, "{}", sym.id);
    out.push(' ');
    out.push_str(&sym.name);

    if let Some(f) = file {
        let rel = f.path.strip_prefix(&index.root).unwrap_or(&f.path);
        let rel = rel.trim_start_matches('/');
        out.push_str("  ");
        out.push_str(rel);
        let _ = write!(out, ":{}-{}", sym.line, sym.end_line);
    } else {
        let _ = write!(out, " :{}-{}", sym.line, sym.end_line);
    }

    if let Some(ref doc) = sym.doc
        && let Some(first) = doc.lines().next()
    {
        let trimmed = first.trim();
        if !trimmed.is_empty() {
            out.push_str(" \"");
            out.push_str(trimmed);
            out.push('"');
        }
    }

    let tree =
        super::tree::AsciiTreeRenderer::new(index).render_callees_with_ids(id, 2, true, false);
    let tree = tree.trim();
    if !tree.is_empty() {
        let mut first = true;
        for line in tree.lines() {
            if first {
                first = false;
                continue;
            }
            out.push_str("\n  ");
            out.push_str(line);
        }
    }

    write_rich_edge_line(out, index, id, EdgeKind::Ref, "\n  > ", max_targets);

    if let Some(parent_id) = sym.parent
        && let Some(sibs) = children_by_parent.get(&parent_id)
    {
        let mut first = true;
        let mut written = 0usize;
        for &cid in sibs {
            if cid == sym.id {
                continue;
            }
            let Some(other) = index.symbols.get(cid as usize) else {
                continue;
            };
            if first {
                out.push_str("\n  ≈ ");
                first = false;
            } else {
                out.push_str(", ");
            }
            out.push_str(&other.name);
            written += 1;
            if written >= max_targets {
                out.push_str(", ...");
                break;
            }
        }
    }
}

fn write_rich_edge_line(
    out: &mut String,
    index: &ProjectIndex,
    id: u32,
    kind: EdgeKind,
    prefix: &str,
    max: usize,
) {
    let Some(edges) = index.forward_edges.get(&id) else {
        return;
    };
    let mut overflow = false;
    for (written, edge) in edges.iter().filter(|e| e.kind == kind).enumerate() {
        if written >= max {
            overflow = true;
            break;
        }
        if written == 0 {
            out.push_str(prefix);
        } else {
            out.push_str(", ");
        }
        match &edge.to {
            Target::Resolved(rid) => {
                if let Some(ts) = index.symbols.get(*rid as usize) {
                    out.push_str(ts.kind.short());
                    out.push('#');
                    let _ = write!(out, "{}", rid);
                    out.push(' ');
                    out.push_str(&ts.name);
                } else {
                    out.push('#');
                    let _ = write!(out, "{}", rid);
                }
            }
            Target::Unresolved(name) | Target::External(name) => {
                out.push('*');
                out.push_str(name);
            }
        }
    }
    if overflow {
        out.push_str(", ...");
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

    pub fn render(&self, file_filter: Option<&[u16]>, kind_filter: Option<&[Kind]>) -> String {
        let kind_set: Option<HashSet<Kind>> = kind_filter.map(|k| k.iter().copied().collect());

        if kind_set.is_some() {
            tracing::debug!("kind_filter active: {:?} kinds", kind_filter);
        }

        let mut out = String::with_capacity(self.max_tokens * crate::render::CHARS_PER_TOKEN);
        out.push_str(AST_HEADER);
        out.push('\n');

        let max_bytes = self.max_tokens * crate::render::CHARS_PER_TOKEN;

        for rel in &self.index.manifest_paths {
            if let Some(content) = load_manifest_content(&self.index.root, rel) {
                let _ = writeln!(out, "@ {}\n{}", rel, content);
                out.push('\n');
            }
            if out.len() >= max_bytes {
                out.push_str("... (truncated)\n");
                return out;
            }
        }

        let children = build_children_by_parent(self.index);

        let filter_set: Option<HashSet<u16>> = file_filter.map(|f| f.iter().copied().collect());

        let mut top_by_file: Vec<Vec<u32>> = (0..self.index.files.len())
            .map(|_| Vec::with_capacity(8))
            .collect();
        for sym in &self.index.symbols {
            if sym.parent.is_none() {
                top_by_file[sym.file_idx as usize].push(sym.id);
            }
        }
        for list in &mut top_by_file {
            list.sort_by_key(|id| {
                self.index
                    .symbols
                    .get(*id as usize)
                    .map(|s| s.line)
                    .unwrap_or(0)
            });
        }

        let mut by_label: BTreeMap<&'static str, (Language, Vec<usize>)> = BTreeMap::new();
        for (file_idx, file) in self.index.files.iter().enumerate() {
            if let Some(ref set) = filter_set
                && !set.contains(&(file_idx as u16))
            {
                continue;
            }
            by_label
                .entry(lang_label(file.language))
                .or_insert_with(|| (file.language, Vec::with_capacity(16)))
                .1
                .push(file_idx);
        }

        for (label, (lang, file_indices)) in &by_label {
            let has_symbols = file_indices.iter().any(|&fi| !top_by_file[fi].is_empty());
            if !has_symbols {
                continue;
            }

            let _ = writeln!(out, "## {}", label);
            out.push_str(lang_legend(*lang));
            out.push('\n');

            for &file_idx in file_indices {
                let top = &top_by_file[file_idx];
                if top.is_empty() {
                    continue;
                }

                let file = &self.index.files[file_idx];
                let rel = file
                    .path
                    .strip_prefix(&self.index.root)
                    .unwrap_or(&file.path)
                    .trim_start_matches('/');
                let _ = writeln!(out, "@ {}", rel);

                for &id in top {
                    if let Some(ref ks) = kind_set
                        && !symbol_or_descendant_matches(self.index, &children, id, ks)
                    {
                        continue;
                    }
                    if render_node(self.index, &children, id, 0, &mut out, *lang, max_bytes) {
                        out.push_str("... (truncated)\n");
                        return out;
                    }
                }
                out.push('\n');
            }
        }

        out
    }
}

pub fn render_files_only(index: &ProjectIndex, file_indices: &[u16]) -> String {
    let mut out = String::with_capacity(file_indices.len() * 128);
    out.push_str("FILES (#N = file id)\n");
    for &idx in file_indices {
        if let Some(file) = index.files.get(idx as usize) {
            let rel = file
                .path
                .strip_prefix(&index.root)
                .unwrap_or(&file.path)
                .trim_start_matches('/');
            let _ = writeln!(out, "# {} @ {}", idx, rel);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use zti_tree_sitter::Language;
    use zti_ts_core::types::Kind;

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
            manifest_paths: Vec::new(),
        }
    }

    fn mk_sym(id: u32, name: &str, kind: Kind, file_idx: u16) -> zti_ts_core::types::Symbol {
        zti_ts_core::types::Symbol {
            id,
            kind,
            name: name.to_string(),
            qualified: name.to_string(),
            file_idx,
            line: id + 1,
            end_line: id + 1,
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
    fn ast_header_lists_every_rust_short_code() {
        for entry in [
            "f=fn",
            "m=method",
            "s=struct",
            "c=class/impl",
            "t=trait",
            "e=enum",
            "v=var",
            "d=mod",
            "k=const",
            "y=typealias",
        ] {
            assert!(
                RUST_LEGEND.contains(entry),
                "RUST_LEGEND missing entry `{}`",
                entry
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
        assert!(
            out.contains("## Rust"),
            "should contain ## Rust section, got:\n{}",
            out
        );
        assert!(out.contains("s#0"), "should contain s#0, got:\n{}", out);
        assert!(
            out.contains("@ a.rs"),
            "should contain @ a.rs, got:\n{}",
            out
        );
    }

    #[test]
    fn struct_with_fields_renders_nested() {
        let mut field = mk_sym(1, "x", Kind::Field, 0);
        field.parent = Some(0);
        let mut field2 = mk_sym(2, "y", Kind::Field, 0);
        field2.parent = Some(0);
        let idx = mk_index(
            vec![mk_sym(0, "Point", Kind::Struct, 0), field, field2],
            vec![mk_file("/p/a.rs")],
        );
        let out = DslRenderer::new(&idx, 8000).render(None, None);
        let section_start = out.find("## Rust").unwrap();
        let section = &out[section_start..];
        let lines: Vec<&str> = section.lines().collect();
        let s_line = lines.iter().find(|l| l.contains("s#0")).unwrap();
        let v1_line = lines.iter().find(|l| l.contains("v#1")).unwrap();
        let v2_line = lines.iter().find(|l| l.contains("v#2")).unwrap();
        assert!(
            s_line.starts_with("s#0"),
            "struct should be at indent 0: {}",
            s_line
        );
        assert!(
            v1_line.starts_with("  v#1"),
            "field should be at indent 1: {}",
            v1_line
        );
        assert!(
            v2_line.starts_with("  v#2"),
            "field should be at indent 1: {}",
            v2_line
        );
    }

    #[test]
    fn impl_emits_separate_symbol_with_methods_nested() {
        let mut method = mk_sym(2, "new", Kind::Method, 0);
        method.parent = Some(1);
        let mut impl_sym = mk_sym(1, "impl State", Kind::Impl, 0);
        impl_sym.signature = "impl State".to_string();
        impl_sym.parent = None;
        let idx = mk_index(
            vec![mk_sym(0, "State", Kind::Struct, 0), impl_sym, method],
            vec![mk_file("/p/a.rs")],
        );
        let out = DslRenderer::new(&idx, 8000).render(None, None);
        let section_start = out.find("## Rust").unwrap();
        let section = &out[section_start..];
        let lines: Vec<&str> = section.lines().collect();
        let c_line = lines.iter().find(|l| l.contains("c#1")).unwrap();
        let m_line = lines.iter().find(|l| l.contains("m#2")).unwrap();
        assert!(
            c_line.contains("impl State"),
            "should contain impl State: {}",
            c_line
        );
        assert!(
            m_line.starts_with("  m#2"),
            "method should be indented under impl: {}",
            m_line
        );
    }

    #[test]
    fn render_files_only_emits_at_paths_without_symbols() {
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

        assert!(
            out.starts_with("FILES"),
            "should start with FILES header, got:\n{}",
            out
        );
        assert!(
            out.contains("# 0 @ "),
            "should contain file ID 0, got:\n{}",
            out
        );
        assert!(
            out.contains("# 1 @ "),
            "should contain file ID 1, got:\n{}",
            out
        );
        assert!(
            out.contains("# 2 @ "),
            "should contain file ID 2, got:\n{}",
            out
        );
        assert!(
            out.contains("src/"),
            "should contain src dir, got:\n{}",
            out
        );
        assert!(
            !out.contains("f#"),
            "should NOT contain symbol DSL, got:\n{}",
            out
        );
        assert!(
            !out.contains("s#"),
            "should NOT contain symbol DSL, got:\n{}",
            out
        );
        assert!(
            !out.contains("foo"),
            "should NOT contain symbol names, got:\n{}",
            out
        );
        assert!(
            out.contains("@ "),
            "should contain @ path prefix, got:\n{}",
            out
        );
    }

    #[test]
    fn format_signature_rust_strips_visibility_and_keywords() {
        assert_eq!(
            format_signature("pub fn foo() -> Result<()>", Language::Rust),
            "foo() -> Result<()>"
        );
        assert_eq!(
            format_signature("fn bar(x: i32)", Language::Rust),
            "bar(x: i32)"
        );
        assert_eq!(format_signature("struct Point {", Language::Rust), "Point");
        assert_eq!(format_signature("enum Color {", Language::Rust), "Color");
        assert_eq!(format_signature("mod utils;", Language::Rust), "utils");
        assert_eq!(
            format_signature("pub(crate) fn hidden()", Language::Rust),
            "hidden()"
        );
        assert_eq!(
            format_signature("const MAX: usize = 10;", Language::Rust),
            "MAX: usize = 10"
        );
        assert_eq!(
            format_signature("pub unsafe async fn combo()", Language::Rust),
            "combo()"
        );
    }

    #[test]
    fn format_signature_ts_strips_chained_keywords() {
        assert_eq!(
            format_signature("public async function foo()", Language::Ts),
            "foo()"
        );
        assert_eq!(
            format_signature("export async function foo()", Language::Ts),
            "foo()"
        );
        assert_eq!(
            format_signature("public static doStuff()", Language::Ts),
            "doStuff()"
        );
        assert_eq!(
            format_signature("export default class Widget {", Language::Tsx),
            "Widget"
        );
    }

    #[test]
    fn format_signature_dart_keeps_return_types() {
        assert_eq!(
            format_signature("Future<List<X>> fetchItems()", Language::Dart),
            "Future<List<X>> fetchItems()"
        );
        assert_eq!(
            format_signature("void main()", Language::Dart),
            "void main()"
        );
        assert_eq!(
            format_signature("static final apiUrl = '...'", Language::Dart),
            "apiUrl = '...'"
        );
        assert_eq!(
            format_signature("abstract class Repo {", Language::Dart),
            "Repo"
        );
    }

    #[test]
    fn format_signature_solidity_strips_chained_modifiers() {
        assert_eq!(
            format_signature("public pure function balanceOf()", Language::Solidity),
            "balanceOf()"
        );
        assert_eq!(
            format_signature("external payable function deposit()", Language::Solidity),
            "deposit()"
        );
        assert_eq!(
            format_signature("uint256 totalSupply = 0", Language::Solidity),
            "uint256 totalSupply = 0"
        );
    }

    fn mk_file_lang(path: &str, lang: Language) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            language: lang,
            imports: HashMap::new(),
        }
    }

    #[test]
    fn render_merges_ts_and_tsx_into_single_section() {
        let rust_sym = mk_sym(0, "Foo", Kind::Struct, 0);
        let ts_sym = mk_sym(1, "Bar", Kind::Class, 1);
        let tsx_sym = mk_sym(2, "Baz", Kind::Class, 2);
        let idx = mk_index(
            vec![rust_sym, ts_sym, tsx_sym],
            vec![
                mk_file_lang("/p/src/main.rs", Language::Rust),
                mk_file_lang("/p/src/app.ts", Language::Ts),
                mk_file_lang("/p/src/view.tsx", Language::Tsx),
            ],
        );
        let out = DslRenderer::new(&idx, 8000).render(None, None);
        let count = out.matches("## TypeScript").count();
        assert_eq!(
            count, 1,
            "Ts and Tsx should merge into one ## TypeScript section, got {} occurrences:\n{}",
            count, out
        );
        assert!(out.contains("s#0"), "should have Rust struct");
        assert!(out.contains("C#1"), "should have TS class");
        assert!(out.contains("C#2"), "should have TSX class");
    }

    #[test]
    fn render_skips_empty_language_sections() {
        let rust_sym = mk_sym(0, "Foo", Kind::Struct, 0);
        let idx = mk_index(vec![rust_sym], vec![mk_file("/p/a.rs")]);
        let out = DslRenderer::new(&idx, 8000).render(None, None);
        assert!(out.contains("## Rust"), "should have Rust section");
        assert!(
            !out.contains("## TypeScript"),
            "should not have empty TypeScript section"
        );
        assert!(
            !out.contains("## Dart"),
            "should not have empty Dart section"
        );
    }

    #[test]
    fn render_with_language_filter_only_emits_that_section() {
        let rust_sym = mk_sym(0, "Foo", Kind::Struct, 0);
        let ts_sym = mk_sym(1, "Bar", Kind::Class, 1);
        let idx = mk_index(
            vec![rust_sym, ts_sym],
            vec![
                mk_file_lang("/p/a.rs", Language::Rust),
                mk_file_lang("/p/b.tsx", Language::Tsx),
            ],
        );
        let filter: Vec<u16> = vec![0];
        let out = DslRenderer::new(&idx, 8000).render(Some(&filter), None);
        assert!(out.contains("## Rust"), "should have Rust section");
        assert!(
            !out.contains("## TypeScript"),
            "should not have TypeScript section when filtered"
        );
        assert!(out.contains("s#0"), "should have Rust struct");
        assert!(!out.contains("C#1"), "should not have TS class");
    }

    #[test]
    fn golden_test_struct_impl_method_nesting() {
        let mut st = mk_sym(0, "DaemonState", Kind::Struct, 0);
        st.signature = "pub struct DaemonState".to_string();
        let mut k = mk_sym(1, "MAX", Kind::Const, 0);
        k.signature = "const MAX: usize = 100;".to_string();
        let mut field = mk_sym(2, "db", Kind::Field, 0);
        field.parent = Some(0);
        field.signature = "db: Arc<Database>".to_string();
        let mut impl_sym = mk_sym(3, "impl DaemonState", Kind::Impl, 0);
        impl_sym.signature = "impl DaemonState".to_string();
        impl_sym.parent = None;
        let mut method = mk_sym(4, "new", Kind::Method, 0);
        method.parent = Some(3);
        method.signature = "pub fn new(p: &Path) -> Self".to_string();

        let idx = mk_index(
            vec![st, k, field, impl_sym, method],
            vec![mk_file("/p/crates/daemon/src/state.rs")],
        );
        let out = DslRenderer::new(&idx, 8000).render(None, None);

        let expected = indoc::indoc! {"
            Code AST. Use node #ID to fetch full code body.

            ## Rust
            Types: @=File, f=fn, m=method, s=struct, c=class/impl, t=trait, e=enum, v=var/field/variant, d=mod, k=const, y=typealias

            @ crates/daemon/src/state.rs
            s#0 DaemonState :1-1
              v#2 db: Arc<Database> :3-3
            k#1 MAX: usize = 100 :2-2
            c#3 impl DaemonState :4-4
              m#4 new(p: &Path) -> Self :5-5

        "};
        assert_eq!(out, expected);
    }
}
