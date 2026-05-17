use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub type ParseResult = Result<(Vec<Symbol>, Vec<Edge>, HashMap<String, String>)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Function => "function",
            Kind::Method => "method",
            Kind::Const => "const",
            Kind::Static => "static",
            Kind::Struct => "struct",
            Kind::Enum => "enum",
            Kind::TypeAlias => "typealias",
            Kind::Class => "class",
            Kind::Interface => "interface",
            Kind::Module => "module",
            Kind::Field => "field",
            Kind::Variant => "variant",
            Kind::Event => "event",
            Kind::Error => "error",
        }
    }

    pub fn from_str_lossy(s: &str) -> Option<Self> {
        match s {
            "fn" | "function" => Some(Kind::Function),
            "method" => Some(Kind::Method),
            "struct" => Some(Kind::Struct),
            "enum" => Some(Kind::Enum),
            "class" => Some(Kind::Class),
            "interface" => Some(Kind::Interface),
            "const" => Some(Kind::Const),
            "static" => Some(Kind::Static),
            "module" | "mod" => Some(Kind::Module),
            "field" => Some(Kind::Field),
            "variant" => Some(Kind::Variant),
            "typealias" | "type" => Some(Kind::TypeAlias),
            "event" => Some(Kind::Event),
            "error" => Some(Kind::Error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    Call,
    Ref,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
