use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use zti_tree_sitter::{Language, detect_from_path, frontend_for};
use zti_ts_core::walker::LanguageFrontend;

use crate::model::{FileEntry, ProjectIndex};

pub(crate) const SKIP_DIRS: &[&str] =
    &[".git", "node_modules", "target", "build", "dist", ".cache"];

pub(crate) const MANIFEST_NAMES: &[&str] =
    &["Cargo.toml", "pubspec.yaml", "package.json", "foundry.toml"];

/// Collect the union of every supported language's `extra_skip_dirs` —
/// the standalone walker has no way to know which languages are present
/// before traversing, so we filter against the superset.
fn all_lang_skip_dirs() -> Vec<&'static str> {
    use zti_tree_sitter::Language;
    let mut out: Vec<&'static str> = Vec::new();
    for lang in [
        Language::Rust,
        Language::Ts,
        Language::Tsx,
        Language::Dart,
        Language::Solidity,
    ] {
        let cfg = zti_tree_sitter::frontend_for(lang).config();
        for &d in cfg.extra_skip_dirs {
            if !out.contains(&d) {
                out.push(d);
            }
        }
    }
    out
}

pub(crate) fn collect_manifest_paths(root: &Path, skip_dirs: Vec<&'static str>) -> Vec<String> {
    let mut results: Vec<String> = Vec::with_capacity(8);
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .filter_entry(move |entry| {
            if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                let name = entry.file_name().to_string_lossy();
                if SKIP_DIRS.contains(&name.as_ref()) || skip_dirs.contains(&name.as_ref()) {
                    return false;
                }
            }
            true
        })
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name() {
            Some(n) => n,
            None => continue,
        };
        if !MANIFEST_NAMES.contains(&name.to_string_lossy().as_ref()) {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
        let rel = rel.trim_start_matches("./");
        results.push(rel.to_string());
    }

    results.sort_by(|a, b| {
        let a_is_root = MANIFEST_NAMES.contains(&a.rsplit('/').next().unwrap_or(a));
        let b_is_root = MANIFEST_NAMES.contains(&b.rsplit('/').next().unwrap_or(b));
        match (a_is_root, b_is_root) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.cmp(b),
        }
    });

    results
}

/// One source file the parser will index. Caller owns the path string (it is
/// moved into the resulting `FileEntry.path`); content is borrowed for the
/// duration of the parse.
pub struct SourceFile<'a> {
    pub full_path: String,
    pub content: &'a str,
    pub language: Language,
}

/// Build a `ProjectIndex` from already-loaded sources. This is the hot path
/// used by `zti-pipeline::indexer::index_project` — the indexer has already
/// walked the filesystem and read every file, so we must not walk a second
/// time.
///
/// `on_progress` is called after each file is parsed with `processed` count,
/// starting from 1.
///
/// When `on_progress` needs the total count, it must be captured from the
/// caller (e.g. by collecting `sources` to a `Vec` first).
pub fn build_index_from_sources<'a, I, F>(
    root: String,
    sources: I,
    on_progress: F,
) -> ProjectIndex
where
    I: IntoIterator<Item = SourceFile<'a>>,
    F: Fn(u32),
{
    let src_iter = sources.into_iter();
    let (lo, _) = src_iter.size_hint();
    let mut files: Vec<FileEntry> = Vec::with_capacity(lo);
    let mut all_symbols: Vec<zti_ts_core::types::Symbol> = Vec::new();
    let mut all_edges: Vec<zti_ts_core::types::Edge> = Vec::new();

    for (i, src) in src_iter.enumerate() {
        let SourceFile {
            full_path,
            content,
            language,
        } = src;
        let file_idx = files.len() as u16;
        let frontend = frontend_for(language);
        let id_offset = all_symbols.len() as u32;

        match frontend.parse(content, file_idx, id_offset) {
            Ok((symbols, edges, imports)) => {
                files.push(FileEntry {
                    path: full_path,
                    language,
                    imports,
                });
                all_symbols.extend(symbols);
                all_edges.extend(edges);
            }
            Err(e) => {
                tracing::warn!("Failed to parse {}: {}", full_path, e);
            }
        }
        on_progress(i as u32 + 1);
    }

    let qualified_map = build_qualified_map(&all_symbols, &files, &root);
    resolve_edges(&mut all_edges, &files, &qualified_map, &all_symbols);

    let reverse_edges = build_reverse_edges(&all_edges);
    let forward_edges = build_forward_edges(&all_edges);

    let root_path = Path::new(&root);
    let skip_dirs = all_lang_skip_dirs();
    let manifest_paths = collect_manifest_paths(root_path, skip_dirs);

    ProjectIndex {
        symbols: all_symbols,
        edges: all_edges,
        files,
        qualified_map,
        reverse_edges,
        forward_edges,
        root,
        manifest_paths,
    }
}

/// Standalone walker entry point — used by `zebra-dsl` and other callers that
/// don't have pre-walked sources. The pipeline must not use this path
/// (it would walk twice); use `build_index_from_sources` instead.
pub fn build_index(root: &str) -> Result<(ProjectIndex, Vec<(String, String)>)> {
    let root_path = Path::new(root).canonicalize()?;
    let root_str = root_path.to_string_lossy().to_string();

    // (full_path, content, language)
    let mut loaded: Vec<(String, String, Language)> = Vec::with_capacity(64);
    let mut text_files: Vec<(String, String)> = Vec::with_capacity(16);

    let walker = WalkBuilder::new(&root_path)
        .hidden(false)
        .git_ignore(true)
        .filter_entry(move |entry| {
            let name = entry.file_name().to_string_lossy();
            if name.starts_with('.') {
                return false;
            }
            if entry.file_type().is_some_and(|ft| ft.is_dir())
                && SKIP_DIRS.contains(&name.as_ref())
            {
                return false;
            }
            true
        })
        .build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let lang = match detect_from_path(path) {
            Some(l) => l,
            None => {
                let path_str = path.to_string_lossy().to_string();
                if let Ok(c) = std::fs::read_to_string(path) {
                    text_files.push((path_str, c));
                }
                continue;
            }
        };
        // Per-file: skip if its language's extra_skip_dirs match any path component
        let skip_dirs = frontend_for(lang).config().extra_skip_dirs;
        if !skip_dirs.is_empty()
            && let Ok(rel) = path.strip_prefix(&root_path)
            && rel.to_string_lossy().split('/').any(|c| skip_dirs.contains(&c))
        {
            continue;
        }
        let path_str = path.to_string_lossy().to_string();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        loaded.push((path_str, content, lang));
    }

    let sources = loaded.iter().map(|(p, c, l)| SourceFile {
        full_path: p.clone(),
        content: c.as_str(),
        language: *l,
    });

    let index = build_index_from_sources(root_str, sources, |_| {});
    Ok((index, text_files))
}

/// Directory names that are source roots (not module names).
fn is_src_root_dir(c: &str) -> bool {
    matches!(c, "src" | "lib" | "app" | "source" | "Sources" | "include")
}

/// File basenames that represent the crate/module root, not a named module.
fn is_non_module_basename(name: &str) -> bool {
    matches!(name, "lib" | "main" | "mod" | "index")
}

fn file_stem(path: &str) -> &str {
    let file_name = match path.rsplit('/').next() {
        Some(name) => name,
        None => path,
    };
    [".rs", ".ts", ".tsx", ".dart", ".sol"]
        .iter()
        .find_map(|suffix| file_name.strip_suffix(suffix))
        .unwrap_or(file_name)
}

fn push_unique_alias(aliases: &mut Vec<String>, alias: String) {
    if !alias.is_empty() && aliases.iter().all(|existing| existing != &alias) {
        aliases.push(alias);
    }
}

fn normalize_rust_path_segment(segment: &str) -> std::borrow::Cow<'_, str> {
    if segment.contains('-') {
        std::borrow::Cow::Owned(segment.replace('-', "_"))
    } else {
        std::borrow::Cow::Borrowed(segment)
    }
}

fn is_rust_non_module_basename(name: &str) -> bool {
    matches!(name, "lib" | "main" | "mod")
}

fn root_crate_name(root: &str) -> Option<std::borrow::Cow<'_, str>> {
    Path::new(root)
        .file_name()
        .and_then(|name| name.to_str())
        .map(normalize_rust_path_segment)
}

fn push_rust_crate_alias(
    aliases: &mut Vec<String>,
    rel: &str,
    root: &str,
    short_path: &str,
    qualified: &str,
) {
    let comp_count = rel.split('/').count();
    let mut comps = Vec::with_capacity(comp_count);
    rel.split('/')
        .filter(|comp| !comp.is_empty())
        .for_each(|comp| comps.push(comp));
    let (crate_name, module_comps) = match comps.as_slice() {
        ["crates", "apps", crate_name, "src", rest @ ..] => (*crate_name, rest),
        ["crates", crate_name, "src", rest @ ..] => (*crate_name, rest),
        ["src", rest @ ..] => match root_crate_name(root) {
            Some(name) => {
                let mut segments = Vec::with_capacity(rest.len().saturating_add(2));
                segments.push(name.into_owned());
                push_rust_module_segments(&mut segments, rest, short_path);
                segments.push(qualified.to_string());
                push_unique_alias(aliases, segments.join("::"));
                return;
            }
            None => return,
        },
        _ => return,
    };

    let mut segments = Vec::with_capacity(comp_count.saturating_add(1));
    segments.push(normalize_rust_path_segment(crate_name).into_owned());
    push_rust_module_segments(&mut segments, module_comps, short_path);
    segments.push(qualified.to_string());
    push_unique_alias(aliases, segments.join("::"));
}

fn push_rust_module_segments(segments: &mut Vec<String>, module_comps: &[&str], short_path: &str) {
    module_comps
        .iter()
        .filter_map(|comp| {
            let stem = file_stem(comp);
            (!stem.is_empty() && !is_rust_non_module_basename(stem))
                .then(|| normalize_rust_path_segment(stem).into_owned())
        })
        .for_each(|segment| segments.push(segment));

    if module_comps.is_empty() && !is_rust_non_module_basename(short_path) {
        segments.push(normalize_rust_path_segment(short_path).into_owned());
    }
}

fn symbol_lookup_aliases(
    sym: &zti_ts_core::types::Symbol,
    file: &FileEntry,
    root: &str,
) -> Vec<String> {
    let file_path = file.path.as_str();
    let short_path = file_stem(file_path);
    let mut aliases = Vec::with_capacity(4);

    // file-basename qualified: keychain::KeyChainErrors
    if short_path != sym.name {
        push_unique_alias(&mut aliases, format!("{}::{}", short_path, sym.name));
    }

    // directory-prefixed qualified: errors::keychain::KeyChainErrors
    // Strip the project root prefix (and trailing slash) to get a relative path.
    let rel = file_path
        .strip_prefix(root)
        .unwrap_or(file_path)
        .trim_start_matches('/');
    if let Some(slash) = rel.rfind('/') {
        let dir_part = &rel[..slash];
        let dir_comp_count = dir_part.split('/').count();
        let mut dir_comps = Vec::with_capacity(dir_comp_count);
        dir_part
            .split('/')
            .filter(|c| !c.is_empty() && !is_src_root_dir(c))
            .for_each(|c| dir_comps.push(c));
        if !dir_comps.is_empty() {
            let dir_prefix = dir_comps.join("::");
            // Always emit the form without the file basename — covers
            // Solidity contracts (where the contract scope is already in
            // sym.qualified) and crate-root files (lib.rs/main.rs).
            push_unique_alias(&mut aliases, format!("{}::{}", dir_prefix, sym.qualified));

            // Also emit the form with the file basename as a module
            // segment, unless the basename is a non-module root
            // (lib/main/mod/index).  This covers Rust files where the
            // file module is not part of sym.qualified.
            if !is_non_module_basename(short_path) {
                push_unique_alias(
                    &mut aliases,
                    format!("{}::{}::{}", dir_prefix, short_path, sym.qualified),
                );
            }
        }
    }

    if file.language == Language::Rust {
        push_rust_crate_alias(&mut aliases, rel, root, short_path, sym.qualified.as_str());
    }

    aliases
}

fn build_qualified_map(
    symbols: &[zti_ts_core::types::Symbol],
    files: &[FileEntry],
    root: &str,
) -> HashMap<String, u32> {
    let mut map = HashMap::with_capacity(symbols.len().saturating_mul(4));

    let mut name_counts: HashMap<&str, usize> = HashMap::with_capacity(symbols.len());
    for sym in symbols {
        *name_counts.entry(sym.name.as_str()).or_insert(0) += 1;
    }

    for sym in symbols {
        let is_unique = name_counts.get(sym.name.as_str()) == Some(&1);
        if sym.qualified != sym.name || is_unique {
            map.entry(sym.qualified.clone()).or_insert(sym.id);
        }

        if let Some(file) = files.get(sym.file_idx as usize) {
            for alias in symbol_lookup_aliases(sym, file, root) {
                map.entry(alias).or_insert(sym.id);
            }
        }

        if is_unique {
            map.entry(sym.name.clone()).or_insert(sym.id);
        }
    }
    map
}

fn resolve_edges(
    edges: &mut [zti_ts_core::types::Edge],
    files: &[FileEntry],
    qualified_map: &HashMap<String, u32>,
    symbols: &[zti_ts_core::types::Symbol],
) {
    for edge in edges.iter_mut() {
        if let zti_ts_core::types::Target::Unresolved(name) = &edge.to {
            let name = name.clone();

            let resolved = if let Some(&id) = qualified_map.get(&name) {
                Some(id)
            } else if let Some(id) =
                resolve_via_imports(&name, edge.from, files, symbols, qualified_map)
            {
                Some(id)
            } else {
                resolve_in_same_file(&name, edge.from, symbols)
            };

            edge.to = match resolved {
                Some(id) => zti_ts_core::types::Target::Resolved(id),
                None => zti_ts_core::types::Target::External(format!("*{}", name)),
            };
        }
    }
}

fn resolve_via_imports(
    name: &str,
    from_id: u32,
    files: &[FileEntry],
    symbols: &[zti_ts_core::types::Symbol],
    qualified_map: &HashMap<String, u32>,
) -> Option<u32> {
    let from_sym = symbols.get(from_id as usize)?;
    let file = files.get(from_sym.file_idx as usize)?;

    if let Some(qualified_path) = file.imports.get(name) {
        if let Some(&id) = qualified_map.get(qualified_path) {
            return Some(id);
        }
        let qualified = format!("{}::{}", qualified_path, name);
        return qualified_map.get(&qualified).copied();
    }

    None
}

fn resolve_in_same_file(
    name: &str,
    from_id: u32,
    symbols: &[zti_ts_core::types::Symbol],
) -> Option<u32> {
    let from_sym = symbols.get(from_id as usize)?;
    symbols
        .iter()
        .find(|s| s.file_idx == from_sym.file_idx && s.name == name)
        .map(|s| s.id)
}

fn build_reverse_edges(
    edges: &[zti_ts_core::types::Edge],
) -> HashMap<u32, Vec<zti_ts_core::types::Edge>> {
    let mut reverse: HashMap<u32, Vec<zti_ts_core::types::Edge>> = HashMap::with_capacity(edges.len());
    for edge in edges {
        if let zti_ts_core::types::Target::Resolved(target_id) = edge.to {
            let reverse_edge = zti_ts_core::types::Edge {
                from: target_id,
                to: zti_ts_core::types::Target::Resolved(edge.from),
                kind: edge.kind,
                line: edge.line,
            };
            reverse.entry(target_id).or_default().push(reverse_edge);
        }
    }
    reverse
}

fn build_forward_edges(
    edges: &[zti_ts_core::types::Edge],
) -> HashMap<u32, Vec<zti_ts_core::types::Edge>> {
    let mut forward: HashMap<u32, Vec<zti_ts_core::types::Edge>> = HashMap::with_capacity(edges.len());
    for edge in edges {
        forward.entry(edge.from).or_default().push(edge.clone());
    }
    forward
}

pub fn files_by_language(files: &[FileEntry], lang: zti_tree_sitter::Language) -> Vec<u16> {
    files
        .iter()
        .enumerate()
        .filter(|(_, f)| f.language == lang)
        .map(|(i, _)| i as u16)
        .collect()
}

pub fn glob_match_files(
    files: &[FileEntry],
    root: &str,
    pattern: &str,
) -> Result<Vec<u16>, String> {
    let glob = globset::Glob::new(pattern)
        .map_err(|e| format!("Invalid glob '{}': {}", pattern, e))?
        .compile_matcher();
    Ok(files
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            let relative = f.path.strip_prefix(root).unwrap_or(&f.path);
            let rel = relative.trim_start_matches('/');
            glob.is_match(rel) || glob.is_match(&f.path)
        })
        .map(|(i, _)| i as u16)
        .collect())
}

pub fn filter_files(
    files: &[FileEntry],
    root: &str,
    glob: Option<&str>,
    lang: Option<Language>,
) -> Result<Vec<u16>, String> {
    let matcher = glob
        .map(|p| {
            globset::Glob::new(p)
                .map_err(|e| format!("Invalid glob '{}': {}", p, e))
                .map(|g| g.compile_matcher())
        })
        .transpose()?;

    Ok(files
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            if let Some(ref g) = matcher {
                let rel = f.path.strip_prefix(root).unwrap_or(&f.path);
                let rel = rel.trim_start_matches('/');
                if !(g.is_match(rel) || g.is_match(&f.path)) {
                    return false;
                }
            }
            if let Some(l) = lang
                && f.language != l
            {
                return false;
            }
            true
        })
        .map(|(i, _)| i as u16)
        .collect())
}

#[cfg(test)]
mod tests {
    use zti_tree_sitter::Language;
    use zti_ts_core::types::{Edge, EdgeKind, Kind, Symbol, Target};

    use crate::model::FileEntry;

    use super::*;

    fn sym(id: u32, name: &str, qualified: &str, file_idx: u16) -> Symbol {
        Symbol {
            id,
            kind: Kind::Method,
            name: name.to_string(),
            qualified: qualified.to_string(),
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

    fn file_entry(idx: u16, path: &str) -> FileEntry {
        let _ = idx;
        FileEntry {
            path: path.to_string(),
            language: Language::Rust,
            imports: HashMap::new(),
        }
    }

    #[test]
    fn ambiguous_bare_name_is_not_in_qualified_map() {
        let symbols = vec![
            sym(0, "parse", "parse", 0),
            sym(1, "parse", "parse", 1),
            sym(2, "parse", "parse", 2),
        ];
        let files = vec![
            file_entry(0, "/p/a.rs"),
            file_entry(1, "/p/b.rs"),
            file_entry(2, "/p/c.rs"),
        ];
        let map = build_qualified_map(&symbols, &files, "/p");
        assert!(
            !map.contains_key("parse"),
            "ambiguous bare name `parse` must not collapse to a single id"
        );
        assert_eq!(map.get("a::parse"), Some(&0));
        assert_eq!(map.get("b::parse"), Some(&1));
        assert_eq!(map.get("c::parse"), Some(&2));
    }

    #[test]
    fn search_dep_candidates_show_resolvable_aliases() {
        let symbols = vec![
            sym(0, "build_index", "build_index", 0),
            sym(1, "build_index", "ChunksTable::build_index", 1),
        ];
        let files = vec![
            file_entry(0, "/p/crates/zti-dsl/src/index.rs"),
            file_entry(1, "/p/crates/zti-store/src/chunks_table.rs"),
        ];
        let qualified_map = build_qualified_map(&symbols, &files, "/p");
        let index = ProjectIndex {
            symbols,
            edges: Vec::with_capacity(0),
            files,
            qualified_map,
            reverse_edges: HashMap::with_capacity(0),
            forward_edges: HashMap::with_capacity(0),
            root: "/p".into(),
            manifest_paths: Vec::with_capacity(0),
        };

        let rendered = crate::search_dep::render_candidates(&index, &[0, 1]);
        assert!(rendered.contains("one of these exact names"));
        assert!(rendered.contains("#0 : method index::build_index"));
        assert!(rendered.contains("#1 : method ChunksTable::build_index"));
        assert!(matches!(
            crate::search_dep::resolve_name(&index, "index::build_index"),
            crate::search_dep::NameMatch::Found(0)
        ));
    }

    #[test]
    fn qualified_map_adds_rust_workspace_crate_aliases() {
        let symbols = vec![
            sym(0, "build_index", "build_index", 0),
            sym(1, "build_index", "ChunksTable::build_index", 1),
            sym(2, "run", "run", 2),
        ];
        let files = vec![
            file_entry(0, "/p/crates/zti-dsl/src/index.rs"),
            file_entry(1, "/p/crates/zti-store/src/chunks_table.rs"),
            file_entry(2, "/p/crates/foo-bar/src/lib.rs"),
        ];
        let map = build_qualified_map(&symbols, &files, "/p");
        assert_eq!(map.get("zti_dsl::index::build_index"), Some(&0));
        assert_eq!(
            map.get("zti_store::chunks_table::ChunksTable::build_index"),
            Some(&1)
        );
        assert_eq!(map.get("foo_bar::run"), Some(&2));
    }

    #[test]
    fn unique_bare_name_resolves() {
        let symbols = vec![sym(0, "only_one", "only_one", 0)];
        let files = vec![file_entry(0, "/p/a.rs")];
        let map = build_qualified_map(&symbols, &files, "/p");
        assert_eq!(map.get("only_one"), Some(&0));
    }

    #[test]
    fn resolve_in_same_file_finds_sibling() {
        let symbols = vec![
            sym(0, "caller", "caller", 0),
            sym(1, "parse", "parse", 0),
            sym(2, "parse", "parse", 1),
        ];
        let got = resolve_in_same_file("parse", 0, &symbols);
        assert_eq!(got, Some(1), "must resolve to the parse in caller's file");
    }

    #[test]
    fn resolve_edges_routes_ambiguous_to_same_file_not_random() {
        let symbols = vec![
            sym(0, "caller", "caller", 0),
            sym(1, "parse", "parse", 0),
            sym(2, "parse", "parse", 1),
            sym(3, "parse", "parse", 2),
        ];
        let files = vec![
            file_entry(0, "/p/a.rs"),
            file_entry(1, "/p/b.rs"),
            file_entry(2, "/p/c.rs"),
        ];
        let mut edges = vec![Edge {
            from: 0,
            to: Target::Unresolved("parse".into()),
            kind: EdgeKind::Call,
            line: 1,
        }];
        let map = build_qualified_map(&symbols, &files, "/p");
        resolve_edges(&mut edges, &files, &map, &symbols);
        match &edges[0].to {
            Target::Resolved(id) => assert_eq!(*id, 1, "should land on same-file parse"),
            other => panic!("expected Resolved(1), got {:?}", other),
        }
    }

    #[test]
    fn resolve_edges_external_when_no_match() {
        let symbols = vec![
            sym(0, "caller", "caller", 0),
            sym(1, "parse", "parse", 1),
            sym(2, "parse", "parse", 2),
        ];
        let files = vec![
            file_entry(0, "/p/a.rs"),
            file_entry(1, "/p/b.rs"),
            file_entry(2, "/p/c.rs"),
        ];
        let mut edges = vec![Edge {
            from: 0,
            to: Target::Unresolved("parse".into()),
            kind: EdgeKind::Call,
            line: 1,
        }];
        let map = build_qualified_map(&symbols, &files, "/p");
        resolve_edges(&mut edges, &files, &map, &symbols);
        assert!(
            matches!(edges[0].to, Target::External(_)),
            "ambiguous, no same-file sibling -> External, got {:?}",
            edges[0].to
        );
    }

    #[test]
    fn files_by_language_returns_correct_indices() {
        let files = vec![
            FileEntry {
                path: "/p/a.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/b.dart".into(),
                language: Language::Dart,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/c.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
        ];
        assert_eq!(files_by_language(&files, Language::Rust), vec![0u16, 2u16]);
        assert_eq!(files_by_language(&files, Language::Dart), vec![1u16]);
        assert!(files_by_language(&files, Language::Ts).is_empty());
    }

    #[test]
    fn glob_match_files_returns_err_on_bad_glob() {
        let files = vec![FileEntry {
            path: "/p/a.rs".into(),
            language: Language::Rust,
            imports: HashMap::new(),
        }];
        assert!(glob_match_files(&files, "/p", "[invalid").is_err());
    }

    #[test]
    fn glob_match_files_matches_paths() {
        let files = vec![
            FileEntry {
                path: "/p/src/a.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/src/b.dart".into(),
                language: Language::Dart,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/lib/c.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
        ];
        let result = glob_match_files(&files, "/p", "src/**/*.rs").unwrap();
        assert_eq!(result, vec![0u16]);
    }

    #[test]
    fn filter_files_combined_glob_and_lang() {
        let files = vec![
            FileEntry {
                path: "/p/src/a.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/src/b.dart".into(),
                language: Language::Dart,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/src/c.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
        ];
        let result = filter_files(&files, "/p", Some("src/**/*.rs"), Some(Language::Rust)).unwrap();
        assert_eq!(result, vec![0u16, 2u16]);
    }

    #[test]
    fn filter_files_glob_only() {
        let files = vec![
            FileEntry {
                path: "/p/src/a.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/lib/b.ts".into(),
                language: Language::Ts,
                imports: HashMap::new(),
            },
        ];
        let result = filter_files(&files, "/p", Some("src/**/*"), None).unwrap();
        assert_eq!(result, vec![0u16]);
    }

    #[test]
    fn filter_files_lang_only() {
        let files = vec![
            FileEntry {
                path: "/p/a.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/b.ts".into(),
                language: Language::Ts,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/c.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
        ];
        let result = filter_files(&files, "/p", None, Some(Language::Rust)).unwrap();
        assert_eq!(result, vec![0u16, 2u16]);
    }

    #[test]
    fn filter_files_bad_glob_returns_err() {
        let files = vec![FileEntry {
            path: "/p/a.rs".into(),
            language: Language::Rust,
            imports: HashMap::new(),
        }];
        assert!(filter_files(&files, "/p", Some("[invalid"), None).is_err());
    }

    #[test]
    fn filter_files_none_returns_all() {
        let files = vec![
            FileEntry {
                path: "/p/a.rs".into(),
                language: Language::Rust,
                imports: HashMap::new(),
            },
            FileEntry {
                path: "/p/b.ts".into(),
                language: Language::Ts,
                imports: HashMap::new(),
            },
        ];
        let result = filter_files(&files, "/p", None, None).unwrap();
        assert_eq!(result, vec![0u16, 1u16]);
    }

    #[test]
    fn resolve_name_finds_unique_bare_name() {
        let symbols = vec![
            sym(0, "parse", "module_a::parse", 0),
            sym(1, "unique_fn", "module_a::unique_fn", 0),
        ];
        let files = vec![file_entry(0, "/p/a.rs")];
        let index = ProjectIndex {
            symbols,
            edges: Vec::new(),
            files,
            qualified_map: build_qualified_map(&[], &[], "/p"),
            reverse_edges: std::collections::HashMap::new(),
            forward_edges: std::collections::HashMap::new(),
            root: "/p".into(),
            manifest_paths: Vec::new(),
        };
        // unique_fn is absent from qualified_map (not in build_qualified_map),
        // but resolve_name should scan and find it.
        match crate::search_dep::resolve_name(&index, "unique_fn") {
            crate::search_dep::NameMatch::Found(id) => assert_eq!(id, 1),
            other => panic!("expected Found, got {:?}", other),
        }
    }

    #[test]
    fn resolve_name_returns_ambiguous_for_duplicated_bare_name() {
        let symbols = vec![
            sym(0, "parse", "parse", 0),
            sym(1, "parse", "parse", 1),
        ];
        let files = vec![file_entry(0, "/p/a.rs"), file_entry(1, "/p/b.rs")];
        let index = ProjectIndex {
            symbols,
            edges: Vec::new(),
            files,
            qualified_map: build_qualified_map(&[], &[], "/p"),
            reverse_edges: std::collections::HashMap::new(),
            forward_edges: std::collections::HashMap::new(),
            root: "/p".into(),
            manifest_paths: Vec::new(),
        };
        match crate::search_dep::resolve_name(&index, "parse") {
            crate::search_dep::NameMatch::Ambiguous(ids) => assert_eq!(ids.len(), 2),
            other => panic!("expected Ambiguous, got {:?}", other),
        }
    }

    #[test]
    fn resolve_name_returns_not_found_for_unknown() {
        let symbols = vec![sym(0, "existing", "existing", 0)];
        let files = vec![file_entry(0, "/p/a.rs")];
        let index = ProjectIndex {
            symbols,
            edges: Vec::new(),
            files,
            qualified_map: build_qualified_map(&[], &[], "/p"),
            reverse_edges: std::collections::HashMap::new(),
            forward_edges: std::collections::HashMap::new(),
            root: "/p".into(),
            manifest_paths: Vec::new(),
        };
        match crate::search_dep::resolve_name(&index, "nonexistent") {
            crate::search_dep::NameMatch::NotFound => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn resolve_name_hits_qualified_map_fast_path() {
        let symbols = vec![sym(0, "Runtime", "tokio::runtime::Runtime", 0)];
        let files = vec![file_entry(0, "/p/lib.rs")];
        let mut qm = std::collections::HashMap::new();
        qm.insert("tokio::runtime::Runtime".to_string(), 0);
        qm.insert("Runtime".to_string(), 0);
        let index = ProjectIndex {
            symbols,
            edges: Vec::new(),
            files,
            qualified_map: qm,
            reverse_edges: std::collections::HashMap::new(),
            forward_edges: std::collections::HashMap::new(),
            root: "/p".into(),
            manifest_paths: Vec::new(),
        };
        match crate::search_dep::resolve_name(&index, "tokio::runtime::Runtime") {
            crate::search_dep::NameMatch::Found(id) => assert_eq!(id, 0),
            other => panic!("expected Found, got {:?}", other),
        }
        match crate::search_dep::resolve_name(&index, "Runtime") {
            crate::search_dep::NameMatch::Found(id) => assert_eq!(id, 0),
            other => panic!("expected Found, got {:?}", other),
        }
    }
}
