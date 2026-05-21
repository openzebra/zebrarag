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
        && let Some(arg) = node.child_by_field_name("argument")
    {
        let path = arg.utf8_text(source.as_bytes()).unwrap_or("");
        let last = path.rsplit("::").next().unwrap_or(path);
        let local = last
            .trim_start_matches('{')
            .trim_start_matches('}')
            .trim()
            .trim_start_matches("self::");
        if !local.is_empty() {
            imports
                .entry(local.to_string())
                .or_insert_with(|| path.to_string());
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

        let impl_sym = symbols.iter().find(|s| s.name == "impl S").unwrap();
        assert_eq!(impl_sym.kind, Kind::Impl);
        assert_eq!(impl_sym.parent, None, "impl S is top-level");

        let foo = symbols.iter().find(|s| s.name == "foo").unwrap();
        assert_eq!(foo.kind, Kind::Method, "foo should be a method, not a fn");
        assert_eq!(
            foo.parent,
            Some(impl_sym.id),
            "foo's parent should be impl S"
        );

        let bar = symbols.iter().find(|s| s.name == "bar").unwrap();
        assert_eq!(bar.kind, Kind::Method);
        assert_eq!(bar.parent, Some(impl_sym.id));
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
        let impl_sym = symbols.iter().find(|s| s.name == "impl E").unwrap();
        assert_eq!(impl_sym.kind, Kind::Impl);
        let name = symbols.iter().find(|s| s.name == "name").unwrap();
        assert_eq!(name.kind, Kind::Method);
        assert_eq!(name.parent, Some(impl_sym.id));
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
        let hi = symbols
            .iter()
            .filter(|s| s.name == "hi" && s.kind == Kind::Method)
            .count();
        assert!(hi >= 1, "trait impl method `hi` should be captured");
        let impl_sym = symbols
            .iter()
            .find(|s| s.name == "impl Greet for T")
            .unwrap();
        let attached = symbols
            .iter()
            .any(|s| s.name == "hi" && s.parent == Some(impl_sym.id));
        assert!(
            attached,
            "trait impl method should have parent = impl symbol"
        );
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
            foo.doc
                .as_ref()
                .is_some_and(|d| d.contains("This is a doc comment")),
            "expected doc comment, got: {:?}",
            foo.doc
        );
    }

    #[test]
    fn rust_impl_emits_separate_symbol_with_methods_nested() {
        let source = indoc::indoc! {"
            struct Foo;
            impl Foo { fn x() {} }
        "};
        let (symbols, _, _) = parse_rust(source);
        let foo = symbols
            .iter()
            .find(|s| s.name == "Foo" && s.kind == Kind::Struct)
            .unwrap();
        assert_eq!(foo.parent, None);
        let impl_sym = symbols.iter().find(|s| s.name == "impl Foo").unwrap();
        assert_eq!(impl_sym.name, "impl Foo");
        assert_eq!(impl_sym.parent, None);
        let x = symbols.iter().find(|s| s.name == "x").unwrap();
        assert_eq!(x.kind, Kind::Method);
        assert_eq!(x.parent, Some(impl_sym.id));
        assert!(
            x.qualified.contains("Foo::x"),
            "qualified should be Foo::x, got: {}",
            x.qualified
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
            foo.doc.is_none()
                || !foo
                    .doc
                    .as_ref()
                    .is_some_and(|d| d.contains("regular comment")),
            "line comment should not be doc, got: {:?}",
            foo.doc
        );
    }

    #[test]
    fn rust_impl_trait_for_type_name_includes_trait() {
        let source = indoc::indoc! {"
            pub struct Foo;
            pub trait Bar { fn baz(&self); }
            impl Bar for Foo {
                fn baz(&self) {}
            }
        "};
        let (symbols, _, _) = parse_rust(source);
        let impl_sym = symbols
            .iter()
            .find(|s| s.name == "impl Bar for Foo")
            .unwrap();
        assert_eq!(impl_sym.kind, Kind::Impl);
        assert_eq!(impl_sym.parent, None);
        let baz = symbols
            .iter()
            .find(|s| s.name == "baz" && s.parent == Some(impl_sym.id))
            .unwrap();
        assert_eq!(baz.kind, Kind::Method);
    }

    #[test]
    fn rust_fn_inside_mod_is_not_retagged_as_method() {
        let source = indoc::indoc! {"
            mod tests {
                fn helper() {}
            }
        "};
        let (symbols, _, _) = parse_rust(source);
        let helper = symbols.iter().find(|s| s.name == "helper").unwrap();
        assert_eq!(
            helper.kind,
            Kind::Function,
            "fn inside mod should stay Function, got {:?}",
            helper.kind
        );
    }

    #[test]
    fn rust_multiline_signature_captured_fully() {
        let source = indoc::indoc! {"
            pub fn bytes_encrypt<R: Rng>(
                rng: &mut R,
                bytes: &[u8],
                pub_key: &PubKey,
            ) -> Result<Vec<u8>, CipherError> {
                todo!()
            }
        "};
        let (symbols, _, _) = parse_rust(source);
        let sym = symbols.iter().find(|s| s.name == "bytes_encrypt").unwrap();
        assert!(
            sym.signature.contains("rng:"),
            "multiline sig should contain args, got: {}",
            sym.signature
        );
        assert!(
            sym.signature.contains("Result<"),
            "multiline sig should contain return type, got: {}",
            sym.signature
        );
        assert!(
            !sym.signature.contains('{'),
            "sig should not contain opening brace, got: {}",
            sym.signature
        );
    }

    #[test]
    fn rust_const_generic_parsed_as_function() {
        let source = indoc::indoc! {r#"
            impl Rq {
                /// Computes the inverse
                pub fn recip<const RATIO: i16>(&self) -> Result<Rq, PolyErrors> {
                    let x = 1;
                    x
                }
            }
        "#};
        let (symbols, _, _) = parse_rust(source);
        let recip = symbols.iter().find(|s| s.name == "recip");
        assert!(recip.is_some(), "recip with const generic should be found, symbols: {symbols:?}");
        let r = recip.unwrap();
        assert_eq!(r.kind, Kind::Method);
        assert!(r.doc.as_ref().is_some_and(|d| d.contains("Computes the inverse")),
            "doc comment should be extracted, got: {:?}", r.doc);
    }
}
