pub mod config;
pub mod types;
pub mod walker;

pub use config::{LangConfig, NameField, extract_name};
pub use types::{Edge, EdgeKind, Import, Kind, ParseResult, Symbol, Target};
pub use walker::{LanguageFrontend, parse_file};
