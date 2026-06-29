use std::collections::HashMap;

use tree_sitter::Node;
use zrag_ts_core::config::{LangConfig, OCAML_CONFIG, OCAML_INTERFACE_CONFIG};
use zrag_ts_core::walker::LanguageFrontend;

pub struct OCamlFrontend {
    pub interface: bool,
}

impl LanguageFrontend for OCamlFrontend {
    fn language(&self) -> tree_sitter::Language {
        if self.interface {
            tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into()
        } else {
            tree_sitter_ocaml::LANGUAGE_OCAML.into()
        }
    }

    fn config(&self) -> &'static LangConfig {
        if self.interface {
            &OCAML_INTERFACE_CONFIG
        } else {
            &OCAML_CONFIG
        }
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::with_capacity(8);
        collect_ocaml_imports(root, source, &mut imports);
        imports
    }
}

fn module_text<'a>(node: Node<'a>, source: &'a str) -> Option<&'a str> {
    node.child_by_field_name("module")
        .and_then(|module| module.utf8_text(source.as_bytes()).ok())
        .map(str::trim)
        .filter(|name| !name.is_empty())
}

fn collect_ocaml_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
    if matches!(node.kind(), "open_module" | "include_module") {
        if let Some(name) = module_text(node, source) {
            let text = node.utf8_text(source.as_bytes()).unwrap_or(name);
            imports
                .entry(name.to_string())
                .or_insert_with(|| text.to_string());
        }
        return;
    }

    let mut cursor = node.walk();
    node.children(&mut cursor)
        .for_each(|child| collect_ocaml_imports(child, source, imports));
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use anyhow::{Result, bail};
    use zrag_ts_core::types::{Edge, Kind, Symbol, Target};

    type FrontendParse = (Vec<Symbol>, Vec<Edge>, HashMap<String, String>);

    use super::*;

    fn parse_ocaml(source: &str) -> Result<FrontendParse> {
        OCamlFrontend { interface: false }.parse(source, 0, 0)
    }

    fn parse_ocaml_interface(source: &str) -> Result<FrontendParse> {
        OCamlFrontend { interface: true }.parse(source, 0, 0)
    }

    #[test]
    fn ocaml_let_function_captured() -> Result<()> {
        let source = "let greet name = name";
        let (symbols, _, _) = parse_ocaml(source)?;
        let greet = symbols.iter().find(|symbol| symbol.name == "greet");
        assert!(matches!(
            greet.map(|symbol| symbol.kind),
            Some(Kind::Function)
        ));
        Ok(())
    }

    #[test]
    fn ocaml_local_paramless_value_skipped_but_local_helper_captured() -> Result<()> {
        let source = indoc::indoc! {"
            let outer p =
              let local_helper q = q + 1 in
              let local_val = 5 in
              local_helper (p + local_val)
        "};
        let (symbols, _, _) = parse_ocaml(source)?;
        let outer = symbols.iter().find(|symbol| symbol.name == "outer");
        assert!(matches!(
            outer.map(|symbol| symbol.kind),
            Some(Kind::Function)
        ));
        let local_helper = symbols.iter().find(|symbol| symbol.name == "local_helper");
        assert!(matches!(
            local_helper.map(|symbol| symbol.kind),
            Some(Kind::Function)
        ));
        assert!(symbols.iter().all(|symbol| symbol.name != "local_val"));
        Ok(())
    }

    #[test]
    fn ocaml_type_captured() -> Result<()> {
        let source = "type person = { name : string }";
        let (symbols, _, _) = parse_ocaml(source)?;
        let person = symbols.iter().find(|symbol| symbol.name == "person");
        assert!(matches!(
            person.map(|symbol| symbol.kind),
            Some(Kind::TypeAlias)
        ));
        Ok(())
    }

    #[test]
    fn ocaml_nested_module_captured() -> Result<()> {
        let source = indoc::indoc! {"
            module Outer = struct
              module Inner = struct
                let value = 1
              end
            end
        "};
        let (symbols, _, _) = parse_ocaml(source)?;
        let inner = symbols.iter().find(|symbol| symbol.name == "Inner");
        assert!(matches!(
            inner.map(|symbol| symbol.kind),
            Some(Kind::Module)
        ));
        assert!(
            symbols
                .iter()
                .any(|symbol| symbol.qualified == "Outer::Inner::value")
        );
        Ok(())
    }

    #[test]
    fn ocaml_class_method_captured() -> Result<()> {
        let source = indoc::indoc! {"
            class greeter = object
              method hello name = name
            end
        "};
        let (symbols, _, _) = parse_ocaml(source)?;
        let hello = symbols.iter().find(|symbol| symbol.name == "hello");
        assert!(matches!(
            hello.map(|symbol| symbol.kind),
            Some(Kind::Method)
        ));
        Ok(())
    }

    #[test]
    fn ocaml_call_edge_captured() -> Result<()> {
        let source = indoc::indoc! {"
            let helper x = x
            let caller y = helper y
        "};
        let (symbols, edges, _) = parse_ocaml(source)?;
        let Some(caller) = symbols.iter().find(|symbol| symbol.name == "caller") else {
            bail!("caller symbol should be indexed");
        };
        let calls = edges.iter().any(|edge| {
            edge.from == caller.id
                && matches!(&edge.to, Target::Unresolved(name) if name == "helper")
        });
        assert!(calls);
        Ok(())
    }

    #[test]
    fn ocaml_open_import_captured() -> Result<()> {
        let source = "open Core";
        let (_, _, imports) = parse_ocaml(source)?;
        assert!(matches!(imports.get("Core"), Some(text) if text == source));
        Ok(())
    }

    #[test]
    fn ocaml_interface_value_specification_captured() -> Result<()> {
        let source = "val greet : string -> string";
        let (symbols, _, _) = parse_ocaml_interface(source)?;
        let greet = symbols.iter().find(|symbol| symbol.name == "greet");
        assert!(matches!(
            greet.map(|symbol| symbol.kind),
            Some(Kind::Function)
        ));
        Ok(())
    }
}
