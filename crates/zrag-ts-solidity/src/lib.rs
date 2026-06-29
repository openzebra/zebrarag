use std::collections::HashMap;

use tree_sitter::Node;
use zrag_ts_core::config::{LangConfig, SOLIDITY_CONFIG};
use zrag_ts_core::walker::LanguageFrontend;

pub struct SolidityFrontend;

impl LanguageFrontend for SolidityFrontend {
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_solidity::LANGUAGE.into()
    }

    fn config(&self) -> &'static LangConfig {
        &SOLIDITY_CONFIG
    }

    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String> {
        let mut imports = HashMap::with_capacity(4);
        collect_solidity_imports(root, source, &mut imports);
        imports
    }
}

fn collect_solidity_imports(node: Node, source: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == "import_directive" {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
        let mut found_named = false;

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "import_declaration" {
                collect_solidity_import_names(child, source, &text, imports, &mut found_named);
            }
        }

        if !found_named {
            let mut cursor2 = node.walk();
            for child in node.children(&mut cursor2) {
                if child.kind() == "identifier"
                    && let Ok(name) = child.utf8_text(source.as_bytes())
                    && !name.is_empty()
                {
                    imports
                        .entry(name.to_string())
                        .or_insert_with(|| text.clone());
                    found_named = true;
                }
            }
        }

        if !found_named && let Some(path_node) = node.child_by_field_name("path") {
            let path_text = path_node.utf8_text(source.as_bytes()).unwrap_or("");
            let clean = path_text.trim_matches('"').trim_matches('\'');
            let basename = clean.rsplit('/').next().unwrap_or(clean);
            let name = basename.trim_end_matches(".sol");
            if !name.is_empty() {
                imports
                    .entry(name.to_string())
                    .or_insert_with(|| text.clone());
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_solidity_imports(child, source, imports);
    }
}

fn collect_solidity_import_names(
    node: Node,
    source: &str,
    text: &str,
    imports: &mut HashMap<String, String>,
    found_named: &mut bool,
) {
    let mut prev_was_as = false;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "as" {
            prev_was_as = true;
            continue;
        }
        if prev_was_as && child.kind() == "identifier" {
            if let Ok(name) = child.utf8_text(source.as_bytes()) {
                imports
                    .entry(name.to_string())
                    .or_insert_with(|| text.to_string());
                *found_named = true;
            }
            prev_was_as = false;
            continue;
        }
        if child.kind() == "identifier"
            && !prev_was_as
            && let Ok(name) = child.utf8_text(source.as_bytes())
        {
            imports
                .entry(name.to_string())
                .or_insert_with(|| text.to_string());
            *found_named = true;
        }
        if child.child_count() > 0 && child.kind() != "identifier" && child.kind() != "as" {
            collect_solidity_import_names(child, source, text, imports, found_named);
        }
    }
}

#[cfg(test)]
mod tests {
    use zrag_ts_core::types::{Edge, Kind, Symbol};

    use super::*;

    fn parse_solidity(source: &str) -> (Vec<Symbol>, Vec<Edge>, HashMap<String, String>) {
        SolidityFrontend.parse(source, 0, 0).unwrap()
    }

    #[test]
    fn solidity_lang_loads() {
        let lang = SolidityFrontend.language();
        assert_ne!(format!("{:?}", lang), "");
    }

    #[test]
    fn contract_captured_as_class() {
        let source = indoc::indoc! {"
            contract Token {
                function transfer(address to, uint256 amount) public {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let token = symbols.iter().find(|s| s.name == "Token").unwrap();
        assert_eq!(token.kind, Kind::Class);
    }

    #[test]
    fn interface_captured() {
        let source = indoc::indoc! {"
            interface IERC20 {
                function transfer(address to, uint256 amount) external;
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let iface = symbols.iter().find(|s| s.name == "IERC20").unwrap();
        assert_eq!(iface.kind, Kind::Interface);
    }

    #[test]
    fn library_captured() {
        let source = indoc::indoc! {"
            library SafeMath {
                function add(uint256 a, uint256 b) internal pure returns (uint256) {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let lib = symbols.iter().find(|s| s.name == "SafeMath").unwrap();
        assert_eq!(lib.kind, Kind::Module);
    }

    #[test]
    fn function_inside_contract_is_method() {
        let source = indoc::indoc! {"
            contract Foo {
                function bar() public {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let bar = symbols.iter().find(|s| s.name == "bar").unwrap();
        assert_eq!(bar.kind, Kind::Method);
    }

    #[test]
    fn free_function_is_function() {
        let source = "function helper() pure {}";
        let (symbols, _, _) = parse_solidity(source);
        let helper = symbols.iter().find(|s| s.name == "helper").unwrap();
        assert_eq!(helper.kind, Kind::Function);
    }

    #[test]
    fn constructor_captured_as_method() {
        let source = indoc::indoc! {"
            contract Foo {
                constructor(address owner) {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let ctor = symbols.iter().find(|s| s.name == "constructor").unwrap();
        assert_eq!(ctor.kind, Kind::Method);
    }

    #[test]
    fn fallback_captured_as_method() {
        let source = indoc::indoc! {"
            contract Foo {
                fallback() external payable {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let fb = symbols.iter().find(|s| s.name == "fallback").unwrap();
        assert_eq!(fb.kind, Kind::Method);
    }

    #[test]
    fn receive_captured_as_method() {
        let source = indoc::indoc! {"
            contract Foo {
                receive() external payable {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let recv = symbols.iter().find(|s| s.name == "receive").unwrap();
        assert_eq!(recv.kind, Kind::Method);
    }

    #[test]
    fn modifier_captured_as_method() {
        let source = indoc::indoc! {"
            contract Foo {
                modifier onlyOwner() { _; }
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let mod_sym = symbols.iter().find(|s| s.name == "onlyOwner").unwrap();
        assert_eq!(mod_sym.kind, Kind::Function);
    }

    #[test]
    fn enum_captured() {
        let source = indoc::indoc! {"
            contract Foo {
                enum Status { Active, Inactive }
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let status = symbols.iter().find(|s| s.name == "Status").unwrap();
        assert_eq!(status.kind, Kind::Enum);
    }

    #[test]
    fn struct_captured() {
        let source = indoc::indoc! {"
            contract Foo {
                struct UserInfo {
                    uint256 balance;
                    bool active;
                }
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let info = symbols.iter().find(|s| s.name == "UserInfo").unwrap();
        assert_eq!(info.kind, Kind::Struct);
    }

    #[test]
    fn state_variable_captured_as_field() {
        let source = indoc::indoc! {"
            contract Foo {
                uint256 public totalSupply;
                address public owner;
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let ts = symbols.iter().find(|s| s.name == "totalSupply").unwrap();
        assert_eq!(ts.kind, Kind::Field);
    }

    #[test]
    fn event_captured() {
        let source = indoc::indoc! {"
            contract Foo {
                event Transfer(address indexed from, address indexed to, uint256 value);
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let evt = symbols.iter().find(|s| s.name == "Transfer").unwrap();
        assert_eq!(evt.kind, Kind::Event);
    }

    #[test]
    fn error_captured() {
        let source = indoc::indoc! {"
            contract Foo {
                error InsufficientBalance(uint256 requested, uint256 available);
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let err = symbols
            .iter()
            .find(|s| s.name == "InsufficientBalance")
            .unwrap();
        assert_eq!(err.kind, Kind::Error);
    }

    #[test]
    fn file_level_constant_captured() {
        let source = indoc::indoc! {"
            uint256 constant MAX_SUPPLY = 1000000;
        "};
        let (symbols, _, _) = parse_solidity(source);
        let max = symbols.iter().find(|s| s.name == "MAX_SUPPLY").unwrap();
        assert_eq!(max.kind, Kind::Const);
    }

    #[test]
    fn user_defined_type_captured() {
        let source = "type Fixed is uint256;";
        let (symbols, _, _) = parse_solidity(source);
        let fixed = symbols.iter().find(|s| s.name == "Fixed").unwrap();
        assert_eq!(fixed.kind, Kind::TypeAlias);
    }

    #[test]
    fn import_named() {
        let source = "import {Foo, Bar} from \"./Token.sol\";";
        let (_, _, imports) = parse_solidity(source);
        assert!(
            imports.contains_key("Foo"),
            "should have Foo, got: {:?}",
            imports
        );
        assert!(
            imports.contains_key("Bar"),
            "should have Bar, got: {:?}",
            imports
        );
    }

    #[test]
    fn import_aliased() {
        let source = "import {Foo as Bar} from \"./Token.sol\";";
        let (_, _, imports) = parse_solidity(source);
        assert!(
            imports.contains_key("Bar"),
            "should have Bar (alias), got: {:?}",
            imports
        );
    }

    #[test]
    fn import_star() {
        let source = "import * as Token from \"./Token.sol\";";
        let (_, _, imports) = parse_solidity(source);
        assert!(
            imports.contains_key("Token"),
            "should have Token, got: {:?}",
            imports
        );
    }

    #[test]
    fn import_bare() {
        let source = "import \"./Token.sol\";";
        let (_, _, imports) = parse_solidity(source);
        assert!(
            imports.is_empty() || imports.contains_key("Token"),
            "bare import should not add noise, got: {:?}",
            imports
        );
    }

    #[test]
    fn doc_comments_not_extracted() {
        let source = indoc::indoc! {"
            contract Foo {
                /// @notice This is a NatSpec doc
                function bar() public {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let bar = symbols.iter().find(|s| s.name == "bar").unwrap();
        assert!(
            bar.doc.is_none(),
            "solidity should not extract docs, got: {:?}",
            bar.doc
        );
    }

    #[test]
    fn line_comments_not_extracted() {
        let source = indoc::indoc! {"
            contract Foo {
                // regular comment before function
                function bar() public {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let bar = symbols.iter().find(|s| s.name == "bar").unwrap();
        assert!(
            bar.doc.is_none(),
            "line comments should not be doc, got: {:?}",
            bar.doc
        );
    }

    #[test]
    fn block_comments_not_extracted() {
        let source = indoc::indoc! {"
            contract Foo {
                /* block comment */
                function bar() public {}
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let bar = symbols.iter().find(|s| s.name == "bar").unwrap();
        assert!(
            bar.doc.is_none(),
            "block comments should not be doc, got: {:?}",
            bar.doc
        );
    }

    #[test]
    fn event_no_doc() {
        let source = indoc::indoc! {"
            contract Foo {
                /// @notice Emitted on transfer
                event Transfer(address from, address to, uint256 amount);
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let evt = symbols.iter().find(|s| s.name == "Transfer").unwrap();
        assert!(
            evt.doc.is_none(),
            "event should not have doc, got: {:?}",
            evt.doc
        );
    }

    #[test]
    fn error_no_doc() {
        let source = indoc::indoc! {"
            contract Foo {
                /// @notice Insufficient balance
                error InsufficientBalance(uint256 requested, uint256 available);
            }
        "};
        let (symbols, _, _) = parse_solidity(source);
        let err = symbols
            .iter()
            .find(|s| s.name == "InsufficientBalance")
            .unwrap();
        assert!(
            err.doc.is_none(),
            "error should not have doc, got: {:?}",
            err.doc
        );
    }
}
