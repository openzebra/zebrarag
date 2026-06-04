use std::collections::HashMap;

use tree_sitter::Node;
use zti_ts_core::config::{LangConfig, TYPESCRIPT_CONFIG};
use zti_ts_core::walker::LanguageFrontend;

pub struct TypeScriptFrontend;

impl LanguageFrontend for TypeScriptFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    }

    fn config(&self) -> &'static LangConfig {
        &TYPESCRIPT_CONFIG
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::with_capacity(8);
        collect_ts_imports(root, source, &mut imports);
        imports
    }
}

fn import_module_stem(import_stmt: Node, source: &str) -> Option<String> {
    const EXTENSIONS: [&str; 6] = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"];

    let src = import_stmt.child_by_field_name("source")?;
    let raw = src.utf8_text(source.as_bytes()).ok()?;
    let spec = raw.trim_matches(|c| matches!(c, '"' | '\'' | '`'));
    let last = match spec.rsplit('/').next() {
        Some(segment) => segment,
        None => spec,
    };
    let stem = match EXTENSIONS
        .iter()
        .find_map(|extension| last.strip_suffix(extension))
    {
        Some(stripped) => stripped,
        None => last,
    };

    (!stem.is_empty() && stem != "." && stem != "..").then(|| stem.to_string())
}

fn collect_ts_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == "import_statement"
        && let Some(module) = import_module_stem(node, source)
    {
        for child in node.children(&mut node.walk()) {
            if child.kind() == "import_clause" {
                for cc in child.children(&mut child.walk()) {
                    if cc.kind() == "identifier"
                        && let Ok(name) = cc.utf8_text(source.as_bytes())
                    {
                        imports
                            .entry(name.to_string())
                            .or_insert_with(|| module.clone());
                    }
                    if cc.kind() == "named_imports" || cc.kind() == "import_list" {
                        for specifier in cc.children(&mut cc.walk()) {
                            if specifier.kind() == "import_specifier" {
                                let key_node = specifier
                                    .child_by_field_name("alias")
                                    .or_else(|| specifier.child_by_field_name("name"));
                                if let Some(kn) = key_node
                                    && let Ok(name) = kn.utf8_text(source.as_bytes())
                                {
                                    imports
                                        .entry(name.to_string())
                                        .or_insert_with(|| module.clone());
                                }
                            }
                        }
                    }
                    if cc.kind() == "namespace_import" {
                        if let Some(id) = cc.child_by_field_name("name") {
                            if let Ok(name) = id.utf8_text(source.as_bytes()) {
                                imports
                                    .entry(name.to_string())
                                    .or_insert_with(|| module.clone());
                            }
                        } else {
                            let mut saw_star = false;
                            for nc in cc.children(&mut cc.walk()) {
                                if nc.kind() == "*" {
                                    saw_star = true;
                                } else if saw_star
                                    && nc.kind() == "identifier"
                                    && let Ok(name) = nc.utf8_text(source.as_bytes())
                                {
                                    imports
                                        .entry(name.to_string())
                                        .or_insert_with(|| module.clone());
                                }
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

#[cfg(test)]
mod tests {
    use zti_ts_core::types::{Edge, Kind, Symbol, Target};

    use super::*;

    fn parse_ts(source: &str) -> (Vec<Symbol>, Vec<Edge>, HashMap<String, String>) {
        TypeScriptFrontend.parse(source, 0, 0).unwrap()
    }

    #[test]
    fn typescript_interface_captured() {
        let source = indoc::indoc! {"
            export interface Greet {
              hi(): void;
              bye(): void;
            }
        "};
        let (symbols, _, _) = parse_ts(source);
        let iface = symbols
            .iter()
            .find(|s| s.name == "Greet")
            .expect("interface Greet missing");
        assert_eq!(iface.kind, Kind::Interface);
    }

    #[test]
    fn no_duplicate_class_from_anonymous_keyword() {
        let source = "class Wallet { constructor() {} toJSON() {} }";
        let (symbols, _, _) = parse_ts(source);
        let class_names: Vec<&str> = symbols
            .iter()
            .filter(|s| s.kind == Kind::Class)
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(class_names, vec!["Wallet"]);
    }

    #[test]
    fn no_duplicate_function_from_anonymous_keyword() {
        let source = "function hello() { return 1; }";
        let (symbols, _, _) = parse_ts(source);
        let fn_names: Vec<&str> = symbols
            .iter()
            .filter(|s| s.kind == Kind::Function)
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(fn_names, vec!["hello"]);
    }

    #[test]
    fn method_definition_detected_as_method() {
        let source = indoc::indoc! {"
            class Foo {
              constructor() {}
              bar() { return 1; }
              static baz() { return 2; }
            }
        "};
        let (symbols, _, _) = parse_ts(source);
        let class = symbols.iter().find(|s| s.kind == Kind::Class).unwrap();
        let methods: Vec<&str> = symbols
            .iter()
            .filter(|s| s.parent == Some(class.id) && s.kind == Kind::Method)
            .map(|s| s.name.as_str())
            .collect();
        assert!(methods.contains(&"constructor"));
        assert!(methods.contains(&"bar"));
        assert!(methods.contains(&"baz"));
    }

    #[test]
    fn private_method_has_clean_name() {
        let source = indoc::indoc! {"
            class Foo {
              #notify() {}
            }
        "};
        let (symbols, _, _) = parse_ts(source);
        let class = symbols.iter().find(|s| s.kind == Kind::Class).unwrap();
        let methods: Vec<&str> = symbols
            .iter()
            .filter(|s| s.parent == Some(class.id) && s.kind == Kind::Method)
            .map(|s| s.name.as_str())
            .collect();
        assert!(methods.contains(&"#notify"));
        assert!(!methods.iter().any(|m| m.contains("const ") || m.len() > 30));
    }

    #[test]
    fn no_anonymous_arrow_functions_as_symbols() {
        let source = indoc::indoc! {"
            const arr = [1, 2, 3];
            arr.map(x => x * 2);
            arr.filter(x => x > 0);
        "};
        let (symbols, _, _) = parse_ts(source);
        let arrow_fns: Vec<&Symbol> = symbols
            .iter()
            .filter(|s| s.kind == Kind::Function && s.name.contains("=>"))
            .collect();
        assert!(
            arrow_fns.is_empty(),
            "anonymous arrow functions should not be symbols, got: {:?}",
            arrow_fns.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn named_arrow_function_in_variable_is_captured() {
        let source = "const myFunc = (x: number) => x + 1;";
        let (symbols, _, _) = parse_ts(source);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"myFunc"));
    }

    #[test]
    fn no_local_variables_inside_methods() {
        let source = indoc::indoc! {"
            class Foo {
              bar() {
                const localVar = 42;
                let anotherVar = 'hello';
              }
            }
        "};
        let (symbols, _, _) = parse_ts(source);
        let class = symbols.iter().find(|s| s.kind == Kind::Class).unwrap();
        let const_names: Vec<&str> = symbols
            .iter()
            .filter(|s| s.parent == Some(class.id) && s.kind == Kind::Const)
            .map(|s| s.name.as_str())
            .collect();
        assert!(
            !const_names.contains(&"localVar"),
            "local variables inside methods should not be class-level const symbols"
        );
    }

    #[test]
    fn module_level_const_is_captured() {
        let source = "const API_URL = 'https://example.com';";
        let (symbols, _, _) = parse_ts(source);
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"API_URL"));
    }

    #[test]
    fn enum_captured() {
        let source = indoc::indoc! {"
            enum Color { Red, Green, Blue }
        "};
        let (symbols, _, _) = parse_ts(source);
        let enum_sym = symbols.iter().find(|s| s.kind == Kind::Enum).unwrap();
        assert_eq!(enum_sym.name, "Color");
    }

    #[test]
    fn function_declaration_edges() {
        let source = indoc::indoc! {"
            function helper() {}
            function add(a: number, b: number) {
              return helper();
            }
        "};
        let (symbols, edges, _) = parse_ts(source);
        let fn_sym = symbols.iter().find(|s| s.name == "add").unwrap();
        assert_eq!(fn_sym.kind, Kind::Function);
        let calls_helper = edges.iter().any(|e| {
            e.from == fn_sym.id && matches!(e.to, Target::Unresolved(ref name) if name == "helper")
        });
        assert!(calls_helper);
    }

    #[test]
    fn typescript_doc_comment_extracted() {
        let source = indoc::indoc! {"
            /** JSDoc description */
            function foo() {}
        "};
        let (symbols, _, _) = parse_ts(source);
        let foo = symbols.iter().find(|s| s.name == "foo").unwrap();
        assert!(
            foo.doc
                .as_ref()
                .is_some_and(|d| d.contains("JSDoc description")),
            "expected JSDoc, got: {:?}",
            foo.doc
        );
    }

    #[test]
    fn named_imports() {
        let source = "import { Foo, Bar } from './module';";
        let (_, _, imports) = parse_ts(source);
        assert_eq!(imports.get("Foo"), Some(&"module".to_string()));
        assert_eq!(imports.get("Bar"), Some(&"module".to_string()));
    }

    #[test]
    fn default_import() {
        let source = "import React from 'react';";
        let (_, _, imports) = parse_ts(source);
        assert_eq!(imports.get("React"), Some(&"react".to_string()));
    }

    #[test]
    fn namespace_import() {
        let source = "import * as fs from 'fs';";
        let (_, _, imports) = parse_ts(source);
        assert_eq!(imports.get("fs"), Some(&"fs".to_string()));
    }

    #[test]
    fn mixed_import() {
        let source = "import React, { useState, useEffect } from 'react';";
        let (_, _, imports) = parse_ts(source);
        assert_eq!(imports.get("React"), Some(&"react".to_string()));
        assert_eq!(imports.get("useState"), Some(&"react".to_string()));
        assert_eq!(imports.get("useEffect"), Some(&"react".to_string()));
    }

    #[test]
    fn aliased_import() {
        let source = "import { Foo as Bar } from './utils';";
        let (_, _, imports) = parse_ts(source);
        assert_eq!(imports.get("Bar"), Some(&"utils".to_string()));
    }

    #[test]
    fn multiple_imports() {
        let source = "import { A } from './a';\nimport { B } from './b';\nimport C from 'c';";
        let (_, _, imports) = parse_ts(source);
        assert!(imports.contains_key("A"));
        assert!(imports.contains_key("B"));
        assert!(imports.contains_key("C"));
    }
}
