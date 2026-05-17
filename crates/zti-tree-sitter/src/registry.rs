use std::collections::HashMap;

use tree_sitter::Node;
use zti_ts_core::config::LangConfig;
use zti_ts_core::types::ParseResult;
use zti_ts_core::walker::LanguageFrontend;
use zti_ts_rust::RustFrontend;
use zti_ts_typescript::TypeScriptFrontend;
use zti_ts_dart::DartFrontend;
use zti_ts_solidity::SolidityFrontend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Ts,
    Tsx,
    Dart,
    Solidity,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Ts => "typescript",
            Language::Tsx => "tsx",
            Language::Dart => "dart",
            Language::Solidity => "solidity",
        }
    }
}

pub enum Frontend {
    Rust(RustFrontend),
    Ts(TypeScriptFrontend),
    Dart(DartFrontend),
    Solidity(SolidityFrontend),
}

impl LanguageFrontend for Frontend {
    fn language(&self) -> tree_sitter::Language {
        match self {
            Frontend::Rust(f) => f.language(),
            Frontend::Ts(f) => f.language(),
            Frontend::Dart(f) => f.language(),
            Frontend::Solidity(f) => f.language(),
        }
    }

    fn config(&self) -> &'static LangConfig {
        match self {
            Frontend::Rust(f) => f.config(),
            Frontend::Ts(f) => f.config(),
            Frontend::Dart(f) => f.config(),
            Frontend::Solidity(f) => f.config(),
        }
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        match self {
            Frontend::Rust(f) => f.extract_imports(root, source),
            Frontend::Ts(f) => f.extract_imports(root, source),
            Frontend::Dart(f) => f.extract_imports(root, source),
            Frontend::Solidity(f) => f.extract_imports(root, source),
        }
    }

    fn parse(&self, source: &str, file_idx: u16, id_start: u32) -> ParseResult {
        match self {
            Frontend::Rust(f) => f.parse(source, file_idx, id_start),
            Frontend::Ts(f) => f.parse(source, file_idx, id_start),
            Frontend::Dart(f) => f.parse(source, file_idx, id_start),
            Frontend::Solidity(f) => f.parse(source, file_idx, id_start),
        }
    }
}

pub fn frontend_for(lang: Language) -> Frontend {
    match lang {
        Language::Rust => Frontend::Rust(RustFrontend),
        Language::Ts | Language::Tsx => Frontend::Ts(TypeScriptFrontend),
        Language::Dart => Frontend::Dart(DartFrontend),
        Language::Solidity => Frontend::Solidity(SolidityFrontend),
    }
}
