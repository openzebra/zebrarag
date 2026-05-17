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

pub fn frontend_for(lang: Language) -> Box<dyn LanguageFrontend> {
    match lang {
        Language::Rust => Box::new(RustFrontend),
        Language::Ts | Language::Tsx => Box::new(TypeScriptFrontend),
        Language::Dart => Box::new(DartFrontend),
        Language::Solidity => Box::new(SolidityFrontend),
    }
}
