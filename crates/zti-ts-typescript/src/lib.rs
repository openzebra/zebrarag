use std::collections::HashMap;

use anyhow::Result;
use tree_sitter::Parser;
use zti_ts_core::config::{LangConfig, TYPESCRIPT_CONFIG};
use zti_ts_core::types::{Edge, Symbol};
use zti_ts_core::walker::{LanguageFrontend, parse_file};

pub struct TypeScriptFrontend;

impl LanguageFrontend for TypeScriptFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    }

    fn config(&self) -> &'static LangConfig {
        &TYPESCRIPT_CONFIG
    }

    fn parse(&self, source: &str, file_idx: u16, id_start: u32) -> Result<(Vec<Symbol>, Vec<Edge>, HashMap<String, String>)> {
        let mut parser = Parser::new();
        parser.set_language(&self.language())?;
        let tree = parser.parse(source, None).ok_or_else(|| anyhow::anyhow!("parse failed"))?;

        let (symbols, edges) = parse_file(&tree, source, file_idx, self.config(), id_start);

        let imports = extract_ts_imports(tree.root_node(), source);

        Ok((symbols, edges, imports))
    }
}

fn extract_ts_imports(node: tree_sitter::Node, source: &str) -> HashMap<String, String> {
    let mut imports = HashMap::new();
    collect_ts_imports(node, source, &mut imports);
    imports
}

fn collect_ts_imports(node: tree_sitter::Node, source: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == "import_statement" {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
        for child in node.children(&mut node.walk()) {
            if child.kind() == "import_clause" {
                for cc in child.children(&mut child.walk()) {
                    if cc.kind() == "identifier"
                        && let Ok(name) = cc.utf8_text(source.as_bytes()) {
                            imports.entry(name.to_string()).or_insert_with(|| text.clone());
                        }
                    if cc.kind() == "named_imports" || cc.kind() == "import_list" {
                        for specifier in cc.children(&mut cc.walk()) {
                            if specifier.kind() == "import_specifier"
                                && let Some(name_node) = specifier.child_by_field_name("name")
                                    && let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                                        imports.entry(name.to_string()).or_insert_with(|| text.clone());
                                    }
                        }
                    }
                }
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_ts_imports(child, source, imports);
    }
}
