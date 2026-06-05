use zti_ts_core::types::Kind;

use crate::registry::Language;

pub fn parse_language(s: &str) -> Option<Language> {
    match s.to_ascii_lowercase().as_str() {
        "rs" | "rust" => Some(Language::Rust),
        "ts" | "typescript" => Some(Language::Ts),
        "tsx" => Some(Language::Tsx),
        "dart" => Some(Language::Dart),
        "sol" | "solidity" => Some(Language::Solidity),
        "py" | "python" => Some(Language::Python),
        "js" | "javascript" => Some(Language::JavaScript),
        "go" => Some(Language::Go),
        "ml" | "ocaml" => Some(Language::OCaml),
        "mli" => Some(Language::OCamlInterface),
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
        assert_eq!(parse_language("py").unwrap(), Language::Python);
        assert_eq!(parse_language("python").unwrap(), Language::Python);
        assert_eq!(parse_language("js").unwrap(), Language::JavaScript);
        assert_eq!(parse_language("javascript").unwrap(), Language::JavaScript);
        assert_eq!(parse_language("go").unwrap(), Language::Go);
        assert_eq!(parse_language("ml").unwrap(), Language::OCaml);
        assert_eq!(parse_language("ocaml").unwrap(), Language::OCaml);
        assert_eq!(parse_language("mli").unwrap(), Language::OCamlInterface);
    }

    #[test]
    fn parse_language_rejects_unknown() {}

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
        assert_eq!(
            kinds,
            vec![Kind::Function, Kind::Struct, Kind::Event, Kind::Error]
        );
    }
}
