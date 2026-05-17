use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use zti_store::FileRow;
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

pub struct Changes {
    pub added: Vec<String>,
    pub modified: Vec<String>,
    pub removed: Vec<String>,
    pub unchanged: Vec<String>,
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
                language: lang.as_str().to_string(),
            },
        );
    }

    map
}

pub fn detect_changes(
    current: &HashMap<String, FileSnapshot>,
    previous: &[FileRow],
) -> Changes {
    let mut prev_map: HashMap<&str, &[u8]> = HashMap::with_capacity(previous.len());
    for row in previous {
        prev_map.insert(&row.file_path, &row.blake3);
    }

    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut unchanged = Vec::new();

    for (rel, snap) in current {
        match prev_map.get(rel.as_str()) {
            Some(&prev_hash) => {
                if snap.blake3.as_slice() == prev_hash {
                    unchanged.push(rel.clone());
                } else {
                    modified.push(rel.clone());
                }
            }
            None => {
                added.push(rel.clone());
            }
        }
    }

    let current_set: HashMap<&str, ()> = current.keys().map(|k| (k.as_str(), ())).collect();
    let removed = previous
        .iter()
        .filter(|row| !current_set.contains_key(row.file_path.as_str()))
        .map(|row| row.file_path.clone())
        .collect();

    Changes {
        added,
        modified,
        removed,
        unchanged,
    }
}
