use std::collections::HashMap;

use zrag_tree_sitter::Language;
use zrag_ts_core::{Edge, Symbol};

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub path: String,
    pub language: Language,
    pub imports: HashMap<String, String>,
}

#[derive(Debug)]
pub struct ProjectIndex {
    pub symbols: Vec<Symbol>,
    pub edges: Vec<Edge>,
    pub files: Vec<FileEntry>,
    pub qualified_map: HashMap<String, u32>,
    pub reverse_edges: HashMap<u32, Vec<Edge>>,
    pub forward_edges: HashMap<u32, Vec<Edge>>,
    pub root: String,
    pub manifest_paths: Vec<String>,
}
