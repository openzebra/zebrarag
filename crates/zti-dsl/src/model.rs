use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    Function,
    Method,
    Const,
    Static,
    Struct,
    Enum,
    TypeAlias,
    Class,
    Interface,
    Module,
    Field,
    Variant,
    Event,
    Error,
}

impl Kind {
    pub fn short(&self) -> &'static str {
        match self {
            Kind::Function => "f",
            Kind::Method => "m",
            Kind::Const => "c",
            Kind::Static => "v",
            Kind::Struct => "s",
            Kind::Enum => "e",
            Kind::TypeAlias => "t",
            Kind::Class => "C",
            Kind::Interface => "I",
            Kind::Module => "M",
            Kind::Field => ".",
            Kind::Variant => ".",
            Kind::Event => "E",
            Kind::Error => "X",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub id: u32,
    pub kind: Kind,
    pub name: String,
    pub qualified: String,
    pub file_idx: u16,
    pub line: u32,
    pub end_line: u32,
    pub signature: String,
    pub doc: Option<String>,
    pub base_classes: Vec<String>,
    pub parent: Option<u32>,
    pub traits: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Target {
    Unresolved(String),
    Resolved(u32),
    External(String),
}

impl Target {
    pub fn display_name(&self) -> &str {
        match self {
            Target::Unresolved(s) => s,
            Target::Resolved(_) => "",
            Target::External(s) => s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Call,
    Ref,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: u32,
    pub to: Target,
    pub kind: EdgeKind,
    pub line: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    TypeScript,
    Dart,
    Solidity,
    Unknown,
}

impl Language {
    pub fn from_path(path: &str) -> Language {
        if path.ends_with(".rs") {
            Language::Rust
        } else if path.ends_with(".ts") || path.ends_with(".tsx") {
            Language::TypeScript
        } else if path.ends_with(".dart") {
            Language::Dart
        } else if path.ends_with(".sol") {
            Language::Solidity
        } else {
            Language::Unknown
        }
    }
}

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
    pub root: String,
}
