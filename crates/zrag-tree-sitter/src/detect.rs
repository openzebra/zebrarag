use std::path::Path;

use crate::registry::Language;

pub fn detect_from_path(path: &Path) -> Option<Language> {
    let ext = path.extension()?.to_str()?;
    match ext {
        "rs" => Some(Language::Rust),
        "ts" => Some(Language::Ts),
        "tsx" => Some(Language::Tsx),
        "dart" => Some(Language::Dart),
        "sol" => Some(Language::Solidity),
        "py" => Some(Language::Python),
        "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "ml" | "scilla" | "scillib" | "scilexp" => Some(Language::OCaml),
        "mli" => Some(Language::OCamlInterface),
        _ => None,
    }
}
