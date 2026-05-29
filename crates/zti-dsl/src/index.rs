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
/// `on_progress` is called after each file is parsed with `(processed, total)`.
pub fn build_index_from_sources<'a, I, F>(
    root: String,
    sources: I,
    on_progress: F,
) -> ProjectIndex
where
    I: IntoIterator<Item = SourceFile<'a>>,
    F: Fn(u32, u32),
{
    let items: Vec<SourceFile<'a>> = sources.into_iter().collect();
    let total = items.len() as u32;
    let mut files: Vec<FileEntry> = Vec::with_capacity(items.len());
    let mut all_symbols: Vec<zti_ts_core::types::Symbol> = Vec::new();
    let mut all_edges: Vec<zti_ts_core::types::Edge> = Vec::new();

    for (i, src) in items.into_iter().enumerate() {
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
        on_progress(i as u32 + 1, total);
    }

    let qualified_map = build_qualified_map(&all_symbols, &files);
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
pub fn build_index(root: &str) -> Result<ProjectIndex> {
    let root_path = Path::new(root).canonicalize()?;

    // (full_path, content, language)
    let mut loaded: Vec<(String, String, Language)> = Vec::new();

    let lang_skip_dirs = all_lang_skip_dirs();
    let walker = WalkBuilder::new(&root_path)
        .hidden(false)
        .git_ignore(true)
        .filter_entry(move |entry| {
            if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                let name = entry.file_name().to_string_lossy();
                if SKIP_DIRS.contains(&name.as_ref()) {
                    return false;
                }
                if lang_skip_dirs.contains(&name.as_ref()) {
                    return false;
                }
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
            None => continue,
        };
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

    Ok(build_index_from_sources(
        root_path.to_string_lossy().to_string(),
        sources,
        |_, _| {},
    ))
}

fn build_qualified_map(
    symbols: &[zti_ts_core::types::Symbol],
    files: &[FileEntry],
) -> HashMap<String, u32> {
    let mut map = HashMap::new();

    let mut name_counts: HashMap<&str, usize> = HashMap::new();
    for sym in symbols {
        *name_counts.entry(sym.name.as_str()).or_insert(0) += 1;
    }

    for sym in symbols {
        let is_unique = name_counts.get(sym.name.as_str()) == Some(&1);
        if sym.qualified != sym.name || is_unique {
            map.entry(sym.qualified.clone()).or_insert(sym.id);
        }

        if let Some(file) = files.get(sym.file_idx as usize) {
            let short_path = file
                .path
                .rsplit('/')
                .next()
                .unwrap_or(&file.path)
                .trim_end_matches(".rs")
                .trim_end_matches(".ts")
                .trim_end_matches(".tsx")
                .trim_end_matches(".dart")
                .to_string();
            if short_path != sym.name {
                let file_qualified = format!("{}::{}", short_path, sym.name);
                map.entry(file_qualified).or_insert(sym.id);
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
    let mut reverse: HashMap<u32, Vec<zti_ts_core::types::Edge>> = HashMap::new();
    for edge in edges {
        if let zti_ts_core::types::Target::Resolved(target_id) = edge.to {
            let mut reverse_edge = edge.clone();
            reverse_edge.from = target_id;
            reverse_edge.to = zti_ts_core::types::Target::Resolved(edge.from);
            reverse.entry(target_id).or_default().push(reverse_edge);
        }
    }
    reverse
}

fn build_forward_edges(
    edges: &[zti_ts_core::types::Edge],
) -> HashMap<u32, Vec<zti_ts_core::types::Edge>> {
    let mut forward: HashMap<u32, Vec<zti_ts_core::types::Edge>> = HashMap::new();
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
        let map = build_qualified_map(&symbols, &files);
        assert!(
            !map.contains_key("parse"),
            "ambiguous bare name `parse` must not collapse to a single id"
        );
        assert_eq!(map.get("a::parse"), Some(&0));
        assert_eq!(map.get("b::parse"), Some(&1));
        assert_eq!(map.get("c::parse"), Some(&2));
    }

    #[test]
    fn unique_bare_name_resolves() {
        let symbols = vec![sym(0, "only_one", "only_one", 0)];
        let files = vec![file_entry(0, "/p/a.rs")];
        let map = build_qualified_map(&symbols, &files);
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
        let map = build_qualified_map(&symbols, &files);
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
        let map = build_qualified_map(&symbols, &files);
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
}
