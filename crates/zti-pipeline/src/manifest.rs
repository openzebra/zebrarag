use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};

use zti_tree_sitter::detect_from_path;

#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub rel_path: String,
    pub mtime_ns: u128,
    pub blake3: [u8; 32],
    pub size_bytes: u64,
    pub contents: String,
    pub language: String,
}

pub fn walk_source_files(root: &Path) -> HashMap<String, FileSnapshot> {
    let mut map = HashMap::new();

    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let lang = match detect_from_path(path) {
            Some(l) => l,
            None => continue,
        };

        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string();

        let contents = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let blake3: [u8; 32] = blake3::hash(contents.as_bytes()).into();

        let rel_path = rel.clone();
        map.insert(
            rel,
            FileSnapshot {
                rel_path,
                mtime_ns,
                blake3,
                size_bytes: metadata.len(),
                contents,
                language: format!("{:?}", lang),
            },
        );
    }

    map
}

pub fn detect_changes(
    current: &HashMap<String, FileSnapshot>,
    previous: &HashSet<String>,
) -> (Vec<String>, Vec<String>) {
    let mut changed = Vec::new();
    let mut unchanged = Vec::new();

    for (rel, snap) in current {
        if previous.contains(rel) {
            unchanged.push(rel.clone());
        } else {
            changed.push(rel.clone());
        }
    }

    (changed, unchanged)
}
