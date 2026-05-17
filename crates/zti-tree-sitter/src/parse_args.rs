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

#[cfg(test)]
mod tests {
    use zti_ts_core::types::Kind;

    use super::*;

    #[test]
    fn parse_language_accepts_known_aliases() {
        assert_eq!(parse_language("rs").unwrap(), Language::Rust);
        assert_eq!(parse_language("rust").unwrap(), Language::Rust);
        assert_eq!(parse_language("RS").unwrap(), Language::Rust);
        assert_eq!(parse_language("ts").unwrap(), Language::Ts);
        assert_eq!(parse_language("typescript").unwrap(), Language::Ts);
        assert_eq!(parse_language("tsx").unwrap(), Language::Tsx);
        assert_eq!(parse_language("dart").unwrap(), Language::Dart);
        assert_eq!(parse_language("sol").unwrap(), Language::Solidity);
        assert_eq!(parse_language("solidity").unwrap(), Language::Solidity);
    }

    #[test]
    fn parse_language_rejects_unknown() {
        assert!(parse_language("js").is_none());
        assert!(parse_language("go").is_none());
        assert!(parse_language("py").is_none());
    }

    #[test]
    fn parse_kinds_returns_empty_when_all_unknown() {
        let input: Vec<String> = vec!["fnn".into(), "strct".into()];
        let kinds = parse_kinds(&input);
        assert!(kinds.is_empty(), "all-unknown should produce empty vec");
    }

    #[test]
    fn parse_kinds_ok_with_mix() {
        let input: Vec<String> = vec!["fn".into(), "bogus".into()];
        let kinds = parse_kinds(&input);
        assert_eq!(kinds, vec![Kind::Function]);
    }

    #[test]
    fn parse_kinds_empty_input_ok() {
        let kinds = parse_kinds(&[]);
        assert!(kinds.is_empty());
    }

    #[test]
    fn parse_kinds_all_known_multi() {
        let input: Vec<String> = vec!["fn".into(), "struct".into(), "event".into(), "error".into()];
        let kinds = parse_kinds(&input);
        assert_eq!(kinds, vec![Kind::Function, Kind::Struct, Kind::Event, Kind::Error]);
    }
}
