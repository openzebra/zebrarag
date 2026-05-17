use zti_ts_core::types::Kind;

use crate::registry::Language;

pub fn parse_language(s: &str) -> Option<Language> {
    match s.to_ascii_lowercase().as_str() {
        "rs" | "rust" => Some(Language::Rust),
        "ts" | "typescript" => Some(Language::Ts),
        "tsx" => Some(Language::Tsx),
        "dart" => Some(Language::Dart),
        "sol" | "solidity" => Some(Language::Solidity),
        _ => None,
    }
}

pub fn parse_kinds(kinds: &[String]) -> Vec<Kind> {
    kinds
        .iter()
        .filter_map(|k| Kind::from_str_lossy(k))
        .collect()
}
