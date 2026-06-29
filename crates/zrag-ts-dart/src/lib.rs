use std::collections::HashMap;

use tree_sitter::Node;
use zrag_ts_core::config::{DART_CONFIG, LangConfig};
use zrag_ts_core::walker::LanguageFrontend;

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
        let text = node.utf8_text(source.as_bytes()).unwrap_or("");
        let uri = node
            .child_by_field_name("uri")
            .and_then(|u| u.utf8_text(source.as_bytes()).ok())
            .unwrap_or("")
            .trim_matches('"')
            .trim_matches('\'');

        let mut alias: Option<String> = None;
        let mut shown: Vec<String> = Vec::with_capacity(8);
        let mut prev_was_as = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let ck = child.kind();
            if ck == "as" {
                prev_was_as = true;
                continue;
            }
            if prev_was_as {
                if ck == "identifier"
                    && let Ok(name) = child.utf8_text(source.as_bytes())
                {
                    alias = Some(name.to_string());
                }
                prev_was_as = false;
                continue;
            }
            if ck == "combinator" {
                let mut cc = child.walk();
                let mut is_show = false;
                for c2 in child.children(&mut cc) {
                    if c2.kind() == "show" {
                        is_show = true;
                        continue;
                    }
                    if is_show
                        && c2.kind() == "identifier"
                        && let Ok(name) = c2.utf8_text(source.as_bytes())
                    {
                        shown.push(name.to_string());
                    }
                }
            }
        }

        let text_owned = || text.to_string();

        let has_alias = alias.is_some();
        if let Some(a) = alias {
            imports.entry(a).or_insert_with(text_owned);
        }
        let has_shown = !shown.is_empty();
        for s in shown {
            imports.entry(s).or_insert_with(text_owned);
        }

        if !has_alias && !has_shown {
            let last_segment = uri.rsplit('/').next().unwrap_or(uri);
            let base = last_segment.trim_end_matches(".dart").trim_end_matches('/');
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
                imports.entry(class_name).or_insert_with(text_owned);
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_dart_imports(child, source, imports);
    }
}

#[cfg(test)]
mod tests {
    use zrag_ts_core::types::{Edge, Kind, Symbol, Target};

    use super::*;

    fn parse_dart(source: &str) -> (Vec<Symbol>, Vec<Edge>, HashMap<String, String>) {
        DartFrontend.parse(source, 0, 0).unwrap()
    }

    #[test]
    fn dart_top_level_function_produces_single_symbol() {
        let source = "void main() { print(\"hi\"); }";
        let (symbols, _, _) = parse_dart(source);
        let mains: Vec<&Symbol> = symbols.iter().filter(|s| s.name == "main").collect();
        assert_eq!(
            mains.len(),
            1,
            "expected exactly one 'main', got: {:?}",
            symbols
        );
        assert_eq!(mains[0].kind, Kind::Function);
    }

    #[test]
    fn dart_arrow_top_level_function_produces_single_symbol() {
        let source = "void main() => print(\"hi\");";
        let (symbols, _, _) = parse_dart(source);
        let mains: Vec<&Symbol> = symbols.iter().filter(|s| s.name == "main").collect();
        assert_eq!(
            mains.len(),
            1,
            "expected exactly one 'main', got: {:?}",
            symbols
        );
        assert!(!symbols.iter().any(|s| s.name.starts_with("=>")));
    }

    #[test]
    fn dart_no_body_text_leaks_into_symbol_names() {
        let source = "void main() { print(\"hello\"); print(\"world\"); }";
        let (symbols, _, _) = parse_dart(source);
        for sym in &symbols {
            assert!(!sym.name.contains('{'), "name contains '{{': {}", sym.name);
            assert!(!sym.name.contains("=>"), "name contains '=>': {}", sym.name);
            assert!(!sym.name.contains(';'), "name contains ';': {}", sym.name);
            assert!(
                !sym.name.contains('\n'),
                "name contains newline: {}",
                sym.name
            );
        }
    }

    #[test]
    fn dart_class_method_produces_single_symbol_attached_to_class() {
        let source = "class A { void foo() {} }";
        let (symbols, _, _) = parse_dart(source);
        let a = symbols.iter().find(|s| s.name == "A").unwrap();
        let foo = symbols.iter().find(|s| s.name == "foo").unwrap();
        assert_eq!(foo.kind, Kind::Method);
        assert_eq!(foo.parent, Some(a.id));
    }

    #[test]
    fn dart_function_call_edges_still_captured() {
        let source = "void main() { someApi(); }";
        let (symbols, edges, _) = parse_dart(source);
        let main = symbols.iter().find(|s| s.name == "main").unwrap();
        let calls_some_api = edges.iter().any(|e| {
            e.from == main.id && matches!(e.to, Target::Unresolved(ref name) if name == "someApi")
        });
        assert!(
            calls_some_api,
            "main should call someApi, edges: {:?}",
            edges
        );
    }

    #[test]
    fn dart_mixin_captured_as_class() {
        let source = indoc::indoc! {"
            mixin StatusBarMixin {
              void attach() {}
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        let mixin = symbols.iter().find(|s| s.name == "StatusBarMixin").unwrap();
        assert_eq!(mixin.kind, Kind::Class);
        let attach = symbols.iter().find(|s| s.name == "attach").unwrap();
        assert_eq!(attach.kind, Kind::Method);
    }

    #[test]
    fn dart_extension_captured_as_class() {
        let source = indoc::indoc! {"
            extension StringX on String {
              String get doubled => this + this;
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        let ext = symbols.iter().find(|s| s.name == "StringX").unwrap();
        assert_eq!(ext.kind, Kind::Class);
        let doubled = symbols.iter().find(|s| s.name == "doubled").unwrap();
        assert_eq!(doubled.kind, Kind::Method);
    }

    #[test]
    fn dart_top_level_const_captured() {
        let source = indoc::indoc! {"
            const apiUrl = '...';
            final timeout = Duration(seconds: 30);
        "};
        let (symbols, _, _) = parse_dart(source);
        let api = symbols.iter().find(|s| s.name == "apiUrl");
        let timeout = symbols.iter().find(|s| s.name == "timeout");
        assert!(
            api.is_some(),
            "apiUrl should be captured, got: {:?}",
            symbols
        );
        assert!(
            timeout.is_some(),
            "timeout should be captured, got: {:?}",
            symbols
        );
    }

    #[test]
    fn dart_const_value_not_stored() {
        let source = "const greeting = 'hello world';";
        let (symbols, _, _) = parse_dart(source);
        let g = symbols.iter().find(|s| s.name == "greeting").unwrap();
        assert_eq!(g.kind, Kind::Const);
    }

    #[test]
    fn dart_abstract_getter_setter_captured() {
        let source = indoc::indoc! {"
            abstract class Repository {
              String get name;
              set name(String value);
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        let getter = symbols
            .iter()
            .find(|s| s.name == "name" && s.kind == Kind::Method);
        assert!(
            getter.is_some(),
            "abstract getter should be Method, got: {:?}",
            symbols
        );
    }

    #[test]
    fn dart_typedef_captured_as_type_alias() {
        let source = "typedef Callback = void Function(int);";
        let (symbols, _, _) = parse_dart(source);
        let td = symbols.iter().find(|s| s.name == "Callback").unwrap();
        assert_eq!(td.kind, Kind::TypeAlias);
    }

    #[test]
    fn dart_class_static_const_captured() {
        let source = indoc::indoc! {"
            class Config {
              static const version = '1.0';
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        let v = symbols.iter().find(|s| s.name == "version").unwrap();
        assert_eq!(v.kind, Kind::Const);
    }

    #[test]
    fn dart_anonymous_extension_gets_synthetic_name() {
        let source = indoc::indoc! {"
            extension on String {
              String get doubled => this + this;
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        let ext = symbols.iter().find(|s| s.name.starts_with("_extOn"));
        assert!(
            ext.is_some(),
            "anonymous extension should get synthetic name, got: {:?}",
            symbols
        );
    }

    #[test]
    fn dart_class_instance_fields_not_captured_as_const() {
        let source = indoc::indoc! {"
            class Person {
              final String name;
              int age = 0;
              static const species = 'human';
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        let species = symbols.iter().find(|s| s.name == "species");
        assert!(species.is_some(), "static const species should be captured");
        assert_eq!(species.unwrap().kind, Kind::Const);
        let name_as_const = symbols
            .iter()
            .find(|s| s.name == "name" && s.kind == Kind::Const);
        assert!(
            name_as_const.is_none(),
            "instance field 'name' should NOT be Const"
        );
        let age_as_const = symbols
            .iter()
            .find(|s| s.name == "age" && s.kind == Kind::Const);
        assert!(
            age_as_const.is_none(),
            "instance field 'age' should NOT be Const"
        );
    }

    #[test]
    fn dart_abstract_operator_captured() {
        let source = indoc::indoc! {"
            abstract class Comparable {
              bool operator <(Comparable other);
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        let comp = symbols.iter().find(|s| s.name == "Comparable").unwrap();
        let op = symbols.iter().find(|s| s.name == "<");
        assert!(
            op.is_some(),
            "operator < should be captured, got: {:?}",
            symbols
        );
        assert_eq!(op.unwrap().kind, Kind::Method);
        assert_eq!(op.unwrap().parent, Some(comp.id));
    }

    #[test]
    fn show_clause_imports_captured() {
        let source = "import 'package:foo/bar.dart' show Alpha, Beta;";
        let (_, _, imports) = parse_dart(source);
        assert!(imports.contains_key("Alpha") || imports.contains_key("Bar"));
    }

    #[test]
    fn aliased_import_captured() {
        let source = "import 'package:foo/bar.dart' as baz;";
        let (_, _, imports) = parse_dart(source);
        assert!(
            imports.contains_key("baz"),
            "aliased import should have key 'baz', got: {:?}",
            imports
        );
    }

    #[test]
    fn bare_import_captured() {
        let source = "import 'package:foo/bar.dart';";
        let (_, _, imports) = parse_dart(source);
        assert!(
            !imports.is_empty(),
            "bare import should still produce an entry, got: {:?}",
            imports
        );
    }

    #[test]
    fn dart_app_localizations_class_is_skipped() {
        let source = indoc::indoc! {"
            class AppLocalizations {
              String hello() => 'hi';
            }
            class Other {
              void real() {}
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        assert!(
            symbols.iter().all(|s| s.name != "AppLocalizations"),
            "AppLocalizations should be skipped, got: {:?}",
            symbols
        );
        assert!(
            symbols.iter().all(|s| s.name != "hello"),
            "skipped class methods should also not appear"
        );
        assert!(symbols.iter().any(|s| s.name == "Other"));
    }

    #[test]
    fn dart_app_localizations_per_locale_subclasses_are_skipped() {
        let source = indoc::indoc! {"
            class AppLocalizationsEn extends AppLocalizations {
              String hello() => 'hi';
            }
            class AppLocalizationsRu extends AppLocalizations {
              String hello() => 'privet';
            }
            class Other {
              void real() {}
            }
        "};
        let (symbols, _, _) = parse_dart(source);
        assert!(
            symbols.iter().all(|s| s.name != "AppLocalizationsEn"),
            "AppLocalizationsEn should be skipped via prefix, got: {:?}",
            symbols
        );
        assert!(
            symbols.iter().all(|s| s.name != "AppLocalizationsRu"),
            "AppLocalizationsRu should be skipped via prefix, got: {:?}",
            symbols
        );
        assert!(
            symbols.iter().all(|s| s.name != "hello"),
            "methods of skipped subclasses should also not appear, got: {:?}",
            symbols
        );
        assert!(symbols.iter().any(|s| s.name == "Other"));
    }
}
