use std::collections::HashMap;

use tree_sitter::Node;
use zti_ts_core::config::{LangConfig, SOLIDITY_CONFIG};
use zti_ts_core::walker::LanguageFrontend;

pub struct SolidityFrontend;

impl LanguageFrontend for SolidityFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_solidity::LANGUAGE.into()
    }

    fn config(&self) -> &'static LangConfig {
        &SOLIDITY_CONFIG
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::new();
        collect_solidity_imports(root, source, &mut imports);
        imports
    }
}

fn collect_solidity_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == "import_directive" {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();

        if let Some(path_node) = node.child_by_field_name("path") {
            let path_text = path_node.utf8_text(source.as_bytes()).unwrap_or("");
            let clean = path_text.trim_matches('"').trim_matches('\'');
            let basename = clean.rsplit('/').next().unwrap_or(clean);
            let name = basename.trim_end_matches(".sol");
            if !name.is_empty() {
                imports.entry(name.to_string()).or_insert_with(|| text.clone());
            }
        }

        for child in node.children(&mut node.walk()) {
            if (child.kind() == "import_declaration" || child.kind() == "identifier")
                && let Ok(name) = child.utf8_text(source.as_bytes())
                    && !name.is_empty() {
                        imports.entry(name.to_string()).or_insert_with(|| text.clone());
                    }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_solidity_imports(child, source, imports);
    }
}
