use std::collections::HashMap;

use anyhow::Result;

pub type ParseResult = Result<(Vec<Symbol>, Vec<Edge>, HashMap<String, String>)>;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone)]
pub struct Import {
    pub local_name: String,
    pub source: String,
}
