use std::collections::HashMap;

use tree_sitter::Node;
use zrag_ts_core::config::{GO_CONFIG, LangConfig};
use zrag_ts_core::walker::LanguageFrontend;

pub struct GoFrontend;

impl LanguageFrontend for GoFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_go::LANGUAGE.into()
    }

    fn config(&self) -> &'static LangConfig {
        &GO_CONFIG
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::with_capacity(8);
        collect_go_imports(root, source, &mut imports);
        imports
    }
}

fn extract_one_spec(
    spec: Node,
    source: &str,
    stmt_text: &str,
    imports: &mut HashMap<String, String>,
) {
    let path = spec
        .child_by_field_name("path")
        .and_then(|p| p.utf8_text(source.as_bytes()).ok())
        .map(|p| p.trim_matches('"').to_string())
        .unwrap_or_default();
    let name = spec
        .child_by_field_name("name")
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())
        .map(|s| s.to_string())
        .or_else(|| {
            let basename = path.rsplit('/').next().unwrap_or(&path);
            Some(basename.to_string())
        });
    if let Some(n) = name
        && !n.is_empty()
    {
        imports.entry(n).or_insert_with(|| stmt_text.to_string());
    }
}

fn collect_go_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == "import_declaration" {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("");
        let mut c = node.walk();
        for child in node.children(&mut c) {
            match child.kind() {
                "import_spec" => {
                    extract_one_spec(child, source, text, imports);
                }
                "import_spec_list" => {
                    let mut c2 = child.walk();
                    for spec in child.children(&mut c2) {
                        if spec.kind() == "import_spec" {
                            extract_one_spec(spec, source, text, imports);
                        }
                    }
                }
                _ => {}
            }
        }
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_go_imports(child, source, imports);
    }
}

#[cfg(test)]
mod tests {
    use zrag_ts_core::types::{Edge, Kind, Symbol, Target};

    use super::*;

    fn parse_go(source: &str) -> (Vec<Symbol>, Vec<Edge>, HashMap<String, String>) {
        GoFrontend.parse(source, 0, 0).unwrap()
    }

    #[test]
    fn go_function_captured() {
        let source = "func greet(name string) string { return name }";
        let (symbols, _, _) = parse_go(source);
        let greet = symbols.iter().find(|s| s.name == "greet").unwrap();
        assert_eq!(greet.kind, Kind::Function);
    }

    #[test]
    fn go_method_captured() {
        let source = indoc::indoc! {"
            type T struct {}
            func (t T) Hello() string { return \"hi\" }
        "};
        let (symbols, _, _) = parse_go(source);
        let hello = symbols.iter().find(|s| s.name == "Hello").unwrap();
        assert_eq!(hello.kind, Kind::Method);
    }

    #[test]
    fn go_call_edge_captured() {
        let source = indoc::indoc! {"
            func helper() int { return 1 }
            func caller() int { return helper() }
        "};
        let (symbols, edges, _) = parse_go(source);
        let caller = symbols.iter().find(|s| s.name == "caller").unwrap();
        let calls = edges.iter().any(|e| {
            e.from == caller.id && matches!(&e.to, Target::Unresolved(name) if name == "helper")
        });
        assert!(calls);
    }

    #[test]
    fn go_import_bare() {
        let source = "import \"fmt\"";
        let (_, _, imports) = parse_go(source);
        assert_eq!(imports.get("fmt"), Some(&source.to_string()));
    }

    #[test]
    fn go_import_grouped() {
        let source = indoc::indoc! {"
            import (
                \"fmt\"
                \"os\"
            )
        "};
        let (_, _, imports) = parse_go(source);
        assert!(imports.contains_key("fmt"));
        assert!(imports.contains_key("os"));
    }

    #[test]
    fn go_import_aliased() {
        let source = "import f \"fmt\"";
        let (_, _, imports) = parse_go(source);
        assert_eq!(imports.get("f"), Some(&source.to_string()));
    }

    #[test]
    fn go_local_var_not_captured() {
        let source = indoc::indoc! {"
            func foo() int {
                x := 42
                return x
            }
        "};
        let (symbols, _, _) = parse_go(source);
        let locals: Vec<&str> = symbols
            .iter()
            .filter(|s| s.name == "x")
            .map(|s| s.name.as_str())
            .collect();
        assert!(locals.is_empty());
    }
}
