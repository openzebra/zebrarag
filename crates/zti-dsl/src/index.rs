use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use zti_tree_sitter::{Language, frontend_for};
use zti_ts_core::walker::LanguageFrontend;

use crate::model::{Edge, EdgeKind, FileEntry, Kind as DslKind, ProjectIndex, Symbol, Target};

fn convert_symbol(s: zti_ts_core::types::Symbol) -> Symbol {
    Symbol {
        id: s.id,
        kind: match s.kind {
            zti_ts_core::types::Kind::Function => DslKind::Function,
            zti_ts_core::types::Kind::Method => DslKind::Method,
            zti_ts_core::types::Kind::Const => DslKind::Const,
            zti_ts_core::types::Kind::Static => DslKind::Static,
            zti_ts_core::types::Kind::Struct => DslKind::Struct,
            zti_ts_core::types::Kind::Enum => DslKind::Enum,
            zti_ts_core::types::Kind::TypeAlias => DslKind::TypeAlias,
            zti_ts_core::types::Kind::Class => DslKind::Class,
            zti_ts_core::types::Kind::Interface => DslKind::Interface,
            zti_ts_core::types::Kind::Module => DslKind::Module,
            zti_ts_core::types::Kind::Field => DslKind::Field,
            zti_ts_core::types::Kind::Variant => DslKind::Variant,
            zti_ts_core::types::Kind::Event => DslKind::Event,
            zti_ts_core::types::Kind::Error => DslKind::Error,
        },
        name: s.name,
        qualified: s.qualified,
        file_idx: s.file_idx,
        line: s.line,
        end_line: s.end_line,
        signature: s.signature,
        doc: s.doc,
        base_classes: s.base_classes,
        parent: s.parent,
        traits: s.traits,
    }
}

fn convert_target(t: zti_ts_core::types::Target) -> Target {
    match t {
        zti_ts_core::types::Target::Unresolved(s) => Target::Unresolved(s),
        zti_ts_core::types::Target::Resolved(id) => Target::Resolved(id),
        zti_ts_core::types::Target::External(s) => Target::External(s),
    }
}

fn convert_edge(e: zti_ts_core::types::Edge) -> Edge {
    Edge {
        from: e.from,
        to: convert_target(e.to),
        kind: match e.kind {
            zti_ts_core::types::EdgeKind::Call => EdgeKind::Call,
            zti_ts_core::types::EdgeKind::Ref => EdgeKind::Ref,
        },
        line: e.line,
    }
}

const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", "build", "dist", ".cache"];
const FORGE_SKIP_DIRS: &[&str] = &["lib"];

pub fn build_index(root: &str) -> Result<ProjectIndex> {
    let root_path = Path::new(root).canonicalize()?;

    let mut files: Vec<FileEntry> = Vec::new();
    let mut all_symbols: Vec<Symbol> = Vec::new();
    let mut all_edges: Vec<Edge> = Vec::new();

    discover_and_parse(&root_path, &mut files, &mut all_symbols, &mut all_edges)?;

    let qualified_map = build_qualified_map(&all_symbols, &files);
    resolve_edges(&mut all_edges, &files, &qualified_map, &all_symbols);

    let reverse_edges = build_reverse_edges(&all_edges);

    Ok(ProjectIndex {
        symbols: all_symbols,
        edges: all_edges,
        files,
        qualified_map,
        reverse_edges,
        root: root_path.to_string_lossy().to_string(),
    })
}

fn discover_and_parse(
    root: &Path,
    files: &mut Vec<FileEntry>,
    all_symbols: &mut Vec<Symbol>,
    all_edges: &mut Vec<Edge>,
) -> Result<()> {
    let is_forge = root.join("foundry.toml").exists();
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .filter_entry(move |entry| {
            if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                let name = entry.file_name().to_string_lossy();
                if SKIP_DIRS.contains(&name.as_ref()) {
                    return false;
                }
                if is_forge && FORGE_SKIP_DIRS.contains(&name.as_ref()) {
                    return false;
                }
            }
            true
        })
        .build();

    let mut file_entries: Vec<(u16, String, crate::model::Language)> = Vec::new();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let path_str = path.to_string_lossy().to_string();
        let lang = crate::model::Language::from_path(&path_str);
        if lang == crate::model::Language::Unknown {
            continue;
        }

        let file_idx = file_entries.len() as u16;
        file_entries.push((file_idx, path_str, lang));
    }

    for (file_idx, path_str, lang) in &file_entries {
        let content = match std::fs::read_to_string(path_str) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let ts_lang = match lang {
            crate::model::Language::Rust => Language::Rust,
            crate::model::Language::TypeScript => Language::Ts,
            crate::model::Language::Dart => Language::Dart,
            crate::model::Language::Solidity => Language::Solidity,
            _ => continue,
        };

        let frontend = frontend_for(ts_lang);
        let id_offset = all_symbols.len() as u32;

        match frontend.parse(&content, *file_idx, id_offset) {
            Ok((symbols, edges, imports)) => {
                files.push(FileEntry {
                    path: path_str.clone(),
                    language: *lang,
                    imports,
                });
                all_symbols.extend(symbols.into_iter().map(convert_symbol));
                all_edges.extend(edges.into_iter().map(convert_edge));
            }
            Err(e) => {
                tracing::warn!("Failed to parse {}: {}", path_str, e);
            }
        }
    }

    Ok(())
}

fn build_qualified_map(symbols: &[Symbol], files: &[FileEntry]) -> HashMap<String, u32> {
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
    edges: &mut [Edge],
    files: &[FileEntry],
    qualified_map: &HashMap<String, u32>,
    symbols: &[Symbol],
) {
    for edge in edges.iter_mut() {
        if let Target::Unresolved(name) = &edge.to {
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
                Some(id) => Target::Resolved(id),
                None => Target::External(format!("*{}", name)),
            };
        }
    }
}

fn resolve_via_imports(
    name: &str,
    from_id: u32,
    files: &[FileEntry],
    symbols: &[Symbol],
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

fn resolve_in_same_file(name: &str, from_id: u32, symbols: &[Symbol]) -> Option<u32> {
    let from_sym = symbols.get(from_id as usize)?;
    symbols
        .iter()
        .find(|s| s.file_idx == from_sym.file_idx && s.name == name)
        .map(|s| s.id)
}

fn build_reverse_edges(edges: &[Edge]) -> HashMap<u32, Vec<Edge>> {
    let mut reverse: HashMap<u32, Vec<Edge>> = HashMap::new();
    for edge in edges {
        if let Target::Resolved(target_id) = edge.to {
            let mut reverse_edge = edge.clone();
            reverse_edge.from = target_id;
            reverse_edge.to = Target::Resolved(edge.from);
            reverse.entry(target_id).or_default().push(reverse_edge);
        }
    }
    reverse
}
