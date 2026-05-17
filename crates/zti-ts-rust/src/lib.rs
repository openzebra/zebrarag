use std::collections::HashMap;

use tree_sitter::Node;
use zti_ts_core::config::{LangConfig, RUST_CONFIG};
use zti_ts_core::walker::LanguageFrontend;

pub struct RustFrontend;

impl LanguageFrontend for RustFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn config(&self) -> &'static LangConfig {
        &RUST_CONFIG
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::new();
        collect_rust_imports(root, source, &mut imports);
        imports
    }
}

fn collect_rust_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
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

#[cfg(test)]
mod tests {
    use zti_ts_core::types::{Edge, Kind, Symbol};

    use super::*;

    fn parse_rust(source: &str) -> (Vec<Symbol>, Vec<Edge>, HashMap<String, String>) {
        RustFrontend.parse(source, 0, 0).unwrap()
    }

    #[test]
    fn rust_impl_method_is_method_with_struct_parent() {
        let source = indoc::indoc! {"
            pub struct S;
            impl S {
                pub fn foo(&self) {}
                fn bar() {}
            }
        "};
        let (symbols, _, _) = parse_rust(source);
        let s = symbols.iter().find(|s| s.name == "S").unwrap();
        assert_eq!(s.kind, Kind::Struct);

        let foo = symbols.iter().find(|s| s.name == "foo").unwrap();
        assert_eq!(foo.kind, Kind::Method, "foo should be a method, not a fn");
        assert_eq!(foo.parent, Some(s.id), "foo's parent should be S");

        let bar = symbols.iter().find(|s| s.name == "bar").unwrap();
        assert_eq!(bar.kind, Kind::Method);
        assert_eq!(bar.parent, Some(s.id));
    }

    #[test]
    fn rust_impl_method_on_enum_attaches_to_enum() {
        let source = indoc::indoc! {"
            pub enum E { A, B }
            impl E {
                pub fn name(&self) -> &str { \"?\" }
            }
        "};
        let (symbols, _, _) = parse_rust(source);
        let e = symbols.iter().find(|s| s.name == "E" && s.kind == Kind::Enum).unwrap();
        let name = symbols.iter().find(|s| s.name == "name").unwrap();
        assert_eq!(name.kind, Kind::Method);
        assert_eq!(name.parent, Some(e.id));
    }

    #[test]
    fn rust_trait_captured_as_interface_with_its_methods() {
        let source = indoc::indoc! {"
            pub trait Greet {
                fn hi(&self);
                fn bye(&self) {}
            }
        "};
        let (symbols, _, _) = parse_rust(source);
        let trait_sym = symbols
            .iter()
            .find(|s| s.name == "Greet")
            .expect("trait Greet missing");
        assert_eq!(trait_sym.kind, Kind::Interface);
        let methods: Vec<&str> = symbols
            .iter()
            .filter(|s| s.parent == Some(trait_sym.id) && s.kind == Kind::Method)
            .map(|s| s.name.as_str())
            .collect();
        assert!(methods.contains(&"hi"), "got: {:?}", methods);
        assert!(methods.contains(&"bye"), "got: {:?}", methods);
    }

    #[test]
    fn rust_trait_impl_does_not_lose_methods() {
        let source = indoc::indoc! {"
            pub struct T;
            pub trait Greet { fn hi(&self); }
            impl Greet for T {
                fn hi(&self) {}
            }
        "};
        let (symbols, _, _) = parse_rust(source);
        let t = symbols.iter().find(|s| s.name == "T" && s.kind == Kind::Struct).unwrap();
        let hi = symbols.iter().filter(|s| s.name == "hi" && s.kind == Kind::Method).count();
        assert!(hi >= 1, "trait impl method `hi` should be captured");
        let attached = symbols.iter().any(|s| s.name == "hi" && s.parent == Some(t.id));
        assert!(attached, "trait impl method should have parent = T");
    }

    #[test]
    fn rust_doc_comment_extracted() {
        let source = indoc::indoc! {"
            /// This is a doc comment
            fn foo() {}
        "};
        let (symbols, _, _) = parse_rust(source);
        let foo = symbols.iter().find(|s| s.name == "foo").unwrap();
        assert!(
            foo.doc.as_ref().is_some_and(|d| d.contains("This is a doc comment")),
            "expected doc comment, got: {:?}",
            foo.doc
        );
    }

    #[test]
    fn rust_line_comment_not_extracted() {
        let source = indoc::indoc! {"
            // regular comment
            fn foo() {}
        "};
        let (symbols, _, _) = parse_rust(source);
        let foo = symbols.iter().find(|s| s.name == "foo").unwrap();
        assert!(
            foo.doc.is_none() || !foo.doc.as_ref().is_some_and(|d| d.contains("regular comment")),
            "line comment should not be doc, got: {:?}",
            foo.doc
        );
    }
}
