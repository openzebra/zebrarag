use std::collections::HashMap;

use tree_sitter::Node;
use zrag_ts_core::config::{LangConfig, PYTHON_CONFIG};
use zrag_ts_core::walker::LanguageFrontend;

pub struct PythonFrontend;

impl LanguageFrontend for PythonFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_python::LANGUAGE.into()
    }

    fn config(&self) -> &'static LangConfig {
        &PYTHON_CONFIG
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::with_capacity(8);
        collect_python_imports(root, source, &mut imports);
        imports
    }
}

fn collect_python_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
    match node.kind() {
        "import_statement" => {
            let text = node.utf8_text(source.as_bytes()).unwrap_or("");
            for child in node.children(&mut node.walk()) {
                if child.kind() == "dotted_name" {
                    let first = child
                        .children(&mut child.walk())
                        .find(|c| c.kind() == "identifier");
                    if let Some(id) = first
                        && let Ok(name) = id.utf8_text(source.as_bytes())
                    {
                        imports.entry(name.to_string()).or_insert(text.to_string());
                    }
                }
                if child.kind() == "aliased_import" {
                    for cc in child.children(&mut child.walk()) {
                        if cc.kind() == "identifier"
                            && let Ok(name) = cc.utf8_text(source.as_bytes())
                        {
                            imports.entry(name.to_string()).or_insert(text.to_string());
                        }
                    }
                }
            }
        }
        "import_from_statement" => {
            let text = node.utf8_text(source.as_bytes()).unwrap_or("");
            // Skip the module_name dotted_name, capture everything else.
            let module_name_id = node.child_by_field_name("module_name").map(|n| n.id());
            let mut c = node.walk();
            for child in node.children(&mut c) {
                let cid = child.id();
                if module_name_id.is_some_and(|mid| mid == cid) {
                    continue;
                }
                match child.kind() {
                    "dotted_name" | "identifier" => {
                        if let Ok(name) = child.utf8_text(source.as_bytes())
                            && !name.is_empty()
                        {
                            imports.entry(name.to_string()).or_insert(text.to_string());
                        }
                    }
                    "aliased_import" => {
                        let mut ac = child.walk();
                        for achild in child.children(&mut ac) {
                            if achild.kind() == "identifier"
                                && let Ok(name) = achild.utf8_text(source.as_bytes())
                            {
                                imports.entry(name.to_string()).or_insert(text.to_string());
                            }
                        }
                    }
                    "wildcard_import" => {
                        imports.entry("*".to_string()).or_insert(text.to_string());
                    }
                    _ => {}
                }
            }
            if let Some(alias) = node.child_by_field_name("alias")
                && let Ok(name) = alias.utf8_text(source.as_bytes())
            {
                imports.entry(name.to_string()).or_insert(text.to_string());
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_python_imports(child, source, imports);
    }
}

#[cfg(test)]
mod tests {
    use zrag_ts_core::types::{Edge, Kind, Symbol, Target};

    use super::*;

    fn parse_py(source: &str) -> (Vec<Symbol>, Vec<Edge>, HashMap<String, String>) {
        PythonFrontend.parse(source, 0, 0).unwrap()
    }

    #[test]
    fn python_function_captured() {
        let source = indoc::indoc! {"
            def greet(name):
                pass
        "};
        let (symbols, _, _) = parse_py(source);
        let greet = symbols.iter().find(|s| s.name == "greet").unwrap();
        assert_eq!(greet.kind, Kind::Function);
    }

    #[test]
    fn python_class_method_is_method() {
        let source = indoc::indoc! {"
            class Greeter:
                def hello(self):
                    pass
                def bye(self):
                    pass
        "};
        let (symbols, _, _) = parse_py(source);
        let cls = symbols.iter().find(|s| s.kind == Kind::Class).unwrap();
        assert_eq!(cls.name, "Greeter");
        let methods: Vec<&str> = symbols
            .iter()
            .filter(|s| s.kind == Kind::Method && s.parent == Some(cls.id))
            .map(|s| s.name.as_str())
            .collect();
        assert!(methods.contains(&"hello"));
        assert!(methods.contains(&"bye"));
    }

    #[test]
    fn python_module_level_const_captured() {
        let source = "MAX_SIZE = 1024";
        let (symbols, _, _) = parse_py(source);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"MAX_SIZE"));
    }

    #[test]
    fn python_local_variable_not_captured() {
        let source = indoc::indoc! {"
            def foo():
                x = 42
                return x
        "};
        let (symbols, _, _) = parse_py(source);
        let locals: Vec<&str> = symbols
            .iter()
            .filter(|s| s.name == "x")
            .map(|s| s.name.as_str())
            .collect();
        assert!(locals.is_empty(), "local x inside fn should not appear");
    }

    #[test]
    fn python_call_edge_captured() {
        let source = indoc::indoc! {"
            def helper():
                pass
            def caller():
                helper()
        "};
        let (symbols, edges, _) = parse_py(source);
        let caller = symbols.iter().find(|s| s.name == "caller").unwrap();
        let calls = edges.iter().any(|e| {
            e.from == caller.id && matches!(&e.to, Target::Unresolved(name) if name == "helper")
        });
        assert!(calls, "caller should have call edge to helper");
    }

    #[test]
    fn python_import_bare() {
        let source = "import os";
        let (_, _, imports) = parse_py(source);
        assert_eq!(imports.get("os"), Some(&"import os".to_string()));
    }

    #[test]
    fn python_import_from() {
        let source = "from os import path";
        let (_, _, imports) = parse_py(source);
        assert_eq!(
            imports.get("path"),
            Some(&"from os import path".to_string())
        );
    }

    #[test]
    fn python_import_aliased() {
        let source = "import numpy as np";
        let (_, _, imports) = parse_py(source);
        assert_eq!(imports.get("np"), Some(&"import numpy as np".to_string()));
    }

    #[test]
    fn python_import_multiple_from() {
        let source = "from os import path, getcwd";
        let (_, _, imports) = parse_py(source);
        assert!(imports.contains_key("path"));
        assert!(imports.contains_key("getcwd"));
    }
}
