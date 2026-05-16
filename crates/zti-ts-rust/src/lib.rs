use std::collections::HashMap;

use anyhow::Result;
use tree_sitter::Parser;
use zti_ts_core::config::{LangConfig, RUST_CONFIG};
use zti_ts_core::types::{Edge, Symbol};
use zti_ts_core::walker::{LanguageFrontend, parse_file};

pub struct RustFrontend;

impl LanguageFrontend for RustFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn config(&self) -> &'static LangConfig {
        &RUST_CONFIG
    }

    fn parse(&self, source: &str, file_idx: u16, id_start: u32) -> Result<(Vec<Symbol>, Vec<Edge>, HashMap<String, String>)> {
        let mut parser = Parser::new();
        parser.set_language(&self.language())?;
        let tree = parser.parse(source, None).ok_or_else(|| anyhow::anyhow!("parse failed"))?;

        let (symbols, edges) = parse_file(&tree, source, file_idx, self.config(), id_start);

        let imports = extract_rust_imports(tree.root_node(), source);

        Ok((symbols, edges, imports))
    }
}

fn extract_rust_imports(node: tree_sitter::Node, source: &str) -> HashMap<String, String> {
    let mut imports = HashMap::new();
    collect_rust_imports(node, source, &mut imports);
    imports
}

fn collect_rust_imports(node: tree_sitter::Node, source: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == "use_declaration"
        && let Some(arg) = node.child_by_field_name("argument") {
            let path = arg.utf8_text(source.as_bytes()).unwrap_or("");
            let last = path.rsplit("::").next().unwrap_or(path);
            let local = last
                .trim_start_matches('{')
                .trim_start_matches('}')
                .trim()
                .trim_start_matches("self::");
            if !local.is_empty() {
                imports.entry(local.to_string()).or_insert_with(|| path.to_string());
            }
        }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_rust_imports(child, source, imports);
    }
}
