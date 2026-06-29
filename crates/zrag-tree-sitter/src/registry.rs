use std::collections::HashMap;

use tree_sitter::Node;
use zrag_ts_core::config::LangConfig;
use zrag_ts_core::types::ParseResult;
use zrag_ts_core::walker::LanguageFrontend;
use zrag_ts_dart::DartFrontend;
use zrag_ts_go::GoFrontend;
use zrag_ts_javascript::JavaScriptFrontend;
use zrag_ts_ocaml::OCamlFrontend;
use zrag_ts_python::PythonFrontend;
use zrag_ts_rust::RustFrontend;
use zrag_ts_solidity::SolidityFrontend;
use zrag_ts_typescript::TypeScriptFrontend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Ts,
    Tsx,
    Dart,
    Solidity,
    Python,
    JavaScript,
    Go,
    OCaml,
    OCamlInterface,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Ts => "typescript",
            Language::Tsx => "tsx",
            Language::Dart => "dart",
            Language::Solidity => "solidity",
            Language::Python => "python",
            Language::JavaScript => "javascript",
            Language::Go => "go",
            Language::OCaml => "ocaml",
            Language::OCamlInterface => "ocaml_interface",
        }
    }
}

pub enum Frontend {
    Rust(RustFrontend),
    Ts(TypeScriptFrontend),
    Dart(DartFrontend),
    Solidity(SolidityFrontend),
    Python(PythonFrontend),
    JavaScript(JavaScriptFrontend),
    Go(GoFrontend),
    OCaml(OCamlFrontend),
}

impl LanguageFrontend for Frontend {
    fn language(&self) -> tree_sitter::Language {
        match self {
            Frontend::Rust(f) => f.language(),
            Frontend::Ts(f) => f.language(),
            Frontend::Dart(f) => f.language(),
            Frontend::Solidity(f) => f.language(),
            Frontend::Python(f) => f.language(),
            Frontend::JavaScript(f) => f.language(),
            Frontend::Go(f) => f.language(),
            Frontend::OCaml(f) => f.language(),
        }
    }

    fn config(&self) -> &'static LangConfig {
        match self {
            Frontend::Rust(f) => f.config(),
            Frontend::Ts(f) => f.config(),
            Frontend::Dart(f) => f.config(),
            Frontend::Solidity(f) => f.config(),
            Frontend::Python(f) => f.config(),
            Frontend::JavaScript(f) => f.config(),
            Frontend::Go(f) => f.config(),
            Frontend::OCaml(f) => f.config(),
        }
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        match self {
            Frontend::Rust(f) => f.extract_imports(root, source),
            Frontend::Ts(f) => f.extract_imports(root, source),
            Frontend::Dart(f) => f.extract_imports(root, source),
            Frontend::Solidity(f) => f.extract_imports(root, source),
            Frontend::Python(f) => f.extract_imports(root, source),
            Frontend::JavaScript(f) => f.extract_imports(root, source),
            Frontend::Go(f) => f.extract_imports(root, source),
            Frontend::OCaml(f) => f.extract_imports(root, source),
        }
    }

    fn parse(&self, source: &str, file_idx: u16, id_start: u32) -> ParseResult {
        match self {
            Frontend::Rust(f) => f.parse(source, file_idx, id_start),
            Frontend::Ts(f) => f.parse(source, file_idx, id_start),
            Frontend::Dart(f) => f.parse(source, file_idx, id_start),
            Frontend::Solidity(f) => f.parse(source, file_idx, id_start),
            Frontend::Python(f) => f.parse(source, file_idx, id_start),
            Frontend::JavaScript(f) => f.parse(source, file_idx, id_start),
            Frontend::Go(f) => f.parse(source, file_idx, id_start),
            Frontend::OCaml(f) => f.parse(source, file_idx, id_start),
        }
    }
}

pub fn frontend_for(lang: Language) -> Frontend {
    match lang {
        Language::Rust => Frontend::Rust(RustFrontend),
        Language::Ts | Language::Tsx => Frontend::Ts(TypeScriptFrontend),
        Language::Dart => Frontend::Dart(DartFrontend),
        Language::Solidity => Frontend::Solidity(SolidityFrontend),
        Language::Python => Frontend::Python(PythonFrontend),
        Language::JavaScript => Frontend::JavaScript(JavaScriptFrontend),
        Language::Go => Frontend::Go(GoFrontend),
        Language::OCaml => Frontend::OCaml(OCamlFrontend { interface: false }),
        Language::OCamlInterface => Frontend::OCaml(OCamlFrontend { interface: true }),
    }
}
