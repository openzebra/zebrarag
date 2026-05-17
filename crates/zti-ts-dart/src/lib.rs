use std::collections::HashMap;

use tree_sitter::Node;
use zti_ts_core::config::{LangConfig, DART_CONFIG};
use zti_ts_core::walker::LanguageFrontend;

pub struct DartFrontend;

impl LanguageFrontend for DartFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_dart::LANGUAGE.into()
    }

    fn config(&self) -> &'static LangConfig {
        &DART_CONFIG
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::new();
        collect_dart_imports(root, source, &mut imports);
        imports
    }
}

fn collect_dart_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == "import_specification" {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
        let uri = node
            .child_by_field_name("uri")
            .and_then(|u| u.utf8_text(source.as_bytes()).ok())
            .unwrap_or("")
            .trim_matches('"')
            .trim_matches('\'');

        let show_clause = node.child_by_field_name("prefix");
        if show_clause.is_some()
            && let Some(name) = show_clause
                && let Ok(n) = name.utf8_text(source.as_bytes()) {
                    imports.entry(n.to_string()).or_insert_with(|| text.clone());
                }

        let last_segment = uri.rsplit('/').next().unwrap_or(uri);
        let base = last_segment
            .trim_end_matches(".dart")
            .trim_end_matches('/');
        if !base.is_empty() {
            let parts: Vec<&str> = base.split('_').collect();
            let class_name: String = parts
                .iter()
                .map(|p| {
                    let mut c = p.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect();
            imports.entry(class_name).or_insert_with(|| text.clone());
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_dart_imports(child, source, imports);
    }
}
