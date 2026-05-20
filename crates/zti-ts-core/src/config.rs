use tree_sitter::Node;

use crate::types::Kind;

#[derive(Debug, Clone)]
pub struct NameField {
    pub node_kind: &'static str,
    pub field: &'static str,
}

#[derive(Debug)]
pub struct TransparentScope {
    pub node_kind: &'static str,
    pub target_field: &'static str,
}

#[derive(Debug, Clone)]
pub struct LangConfig {
    pub scope_nodes: &'static [&'static str],
    pub name_fields: &'static [NameField],
    pub kind_map: &'static [(&'static str, Kind)],
    pub container_kinds: &'static [Kind],
    pub call_nodes: &'static [&'static str],
    pub call_field: &'static str,
    pub ref_node: &'static str,
    pub ref_field: &'static str,
    pub import_node: &'static str,
    pub extra_skip_dirs: &'static [&'static str],
    pub transparent_scope_kinds: &'static [TransparentScope],
    pub extract_docs: bool,
    pub instance_field_kinds: &'static [&'static str],
    pub no_retag_kinds: &'static [&'static str],
    pub impl_node: Option<&'static str>,
    pub symbol_name_skip: &'static [&'static str],
    pub symbol_name_skip_prefix: &'static [&'static str],
}

impl LangConfig {
    pub fn kind_for(&self, node: &Node) -> Option<Kind> {
        let kind_str = node.kind();
        for &(pattern, kind) in self.kind_map {
            if pattern == kind_str {
                return Some(kind);
            }
        }
        None
    }

    pub fn is_scope(&self, node: &Node) -> bool {
        self.scope_nodes.contains(&node.kind())
    }
}

fn read_field(node: &Node, source: &str, field: &str) -> Option<String> {
    let child = node.child_by_field_name(field)?;
    let text = child.utf8_text(source.as_bytes()).ok()?;
    (!text.is_empty()).then(|| text.to_string())
}

fn first_named_identifier(node: &Node, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if matches!(
            child.kind(),
            "identifier"
                | "type_identifier"
                | "field_identifier"
                | "property_identifier"
                | "private_property_identifier"
        ) && let Ok(text) = child.utf8_text(source.as_bytes())
            && !text.is_empty()
        {
            return Some(text.to_string());
        }
    }
    None
}

pub fn extract_name(node: &Node, source: &str, config: &LangConfig) -> Option<String> {
    for field_def in config.name_fields {
        if node.kind() == field_def.node_kind
            && let Some(name) = read_field(node, source, field_def.field)
        {
            return Some(name);
        }
    }

    let kind = node.kind();

    if kind == "extension_declaration" {
        if let Some(name) = read_field(node, source, "name") {
            return Some(name);
        }
        let mut on_type = None;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "on" {
                on_type = Some(child);
            } else if on_type.is_some()
                && let Ok(text) = child.utf8_text(source.as_bytes())
                && !text.is_empty()
            {
                return Some(format!("_extOn{}", text));
            }
        }
    }

    if kind == "operator_signature"
        && let Some(op) = read_field(node, source, "operator")
    {
        return Some(op);
    }

    if matches!(
        kind,
        "function_declaration" | "method_declaration" | "method_signature"
    ) {
        let sig = node.child_by_field_name("signature").or_else(|| {
            let mut c = node.walk();
            node.children(&mut c)
                .find(|ch| matches!(ch.kind(), "function_signature" | "method_signature"))
        });
        if let Some(sig) = sig
            && let Some(name) = extract_name(&sig, source, config)
        {
            return Some(name);
        }
    }

    if kind == "constructor_definition" {
        return Some("constructor".to_string());
    }

    if kind == "fallback_receive_definition" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let ck = child.kind();
            if (ck == "fallback" || ck == "receive")
                && let Ok(text) = child.utf8_text(source.as_bytes())
            {
                return Some(text.to_string());
            }
        }
    }

    if kind == "variable_declarator"
        && let Some(name) = read_field(node, source, "name")
    {
        return Some(name);
    }

    if matches!(
        kind,
        "top_level_variable_declaration" | "initialized_identifier" | "static_final_declaration"
    ) {
        if let Some(name) = read_field(node, source, "name") {
            return Some(name);
        }
        if kind == "initialized_identifier"
            && let Some(name) = first_named_identifier(node, source)
        {
            return Some(name);
        }
    }

    first_named_identifier(node, source)
}

pub static RUST_CONFIG: LangConfig = LangConfig {
    scope_nodes: &[
        "function_item",
        "impl_item",
        "trait_item",
        "mod_item",
        "enum_item",
        "struct_item",
        "type_item",
        "const_item",
        "static_item",
        "block",
    ],
    name_fields: &[
        NameField {
            node_kind: "function_item",
            field: "name",
        },
        NameField {
            node_kind: "struct_item",
            field: "name",
        },
        NameField {
            node_kind: "enum_item",
            field: "name",
        },
        NameField {
            node_kind: "trait_item",
            field: "name",
        },
        NameField {
            node_kind: "mod_item",
            field: "name",
        },
        NameField {
            node_kind: "type_item",
            field: "name",
        },
        NameField {
            node_kind: "const_item",
            field: "name",
        },
        NameField {
            node_kind: "static_item",
            field: "name",
        },
        NameField {
            node_kind: "field_item",
            field: "name",
        },
        NameField {
            node_kind: "enum_variant",
            field: "name",
        },
        NameField {
            node_kind: "function_signature_item",
            field: "name",
        },
        NameField {
            node_kind: "metavariable",
            field: "name",
        },
        NameField {
            node_kind: "macro_invocation",
            field: "name",
        },
    ],
    kind_map: &[
        ("function_item", Kind::Function),
        ("function_signature_item", Kind::Function),
        ("struct_item", Kind::Struct),
        ("enum_item", Kind::Enum),
        ("enum_variant", Kind::Variant),
        ("trait_item", Kind::Interface),
        ("mod_item", Kind::Module),
        ("type_item", Kind::TypeAlias),
        ("const_item", Kind::Const),
        ("static_item", Kind::Static),
        ("field_item", Kind::Field),
        ("impl_item", Kind::Impl),
        ("macro_definition", Kind::Function),
        ("metavariable", Kind::Const),
    ],
    container_kinds: &[
        Kind::Struct,
        Kind::Enum,
        Kind::Interface,
        Kind::Module,
        Kind::Class,
        Kind::Impl,
    ],
    call_nodes: &["call_expression", "macro_invocation"],
    call_field: "function",
    ref_node: "scoped_identifier",
    ref_field: "name",
    import_node: "use_declaration",
    extra_skip_dirs: &[],
    transparent_scope_kinds: &[],
    extract_docs: true,
    instance_field_kinds: &[],
    no_retag_kinds: &[],
    impl_node: Some("impl_item"),
    symbol_name_skip: &[],
    symbol_name_skip_prefix: &[],
};

pub static TYPESCRIPT_CONFIG: LangConfig = LangConfig {
    scope_nodes: &[
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
        "method_definition",
        "arrow_function",
        "program",
        "export_statement",
        "lexical_declaration",
        "variable_declaration",
    ],
    name_fields: &[
        NameField {
            node_kind: "function_declaration",
            field: "name",
        },
        NameField {
            node_kind: "class_declaration",
            field: "name",
        },
        NameField {
            node_kind: "interface_declaration",
            field: "name",
        },
        NameField {
            node_kind: "enum_declaration",
            field: "name",
        },
        NameField {
            node_kind: "type_alias_declaration",
            field: "name",
        },
        NameField {
            node_kind: "method_definition",
            field: "name",
        },
        NameField {
            node_kind: "public_field_definition",
            field: "name",
        },
        NameField {
            node_kind: "enum_assignment",
            field: "name",
        },
        NameField {
            node_kind: "property_identifier",
            field: "value",
        },
    ],
    kind_map: &[
        ("function_declaration", Kind::Function),
        ("class_declaration", Kind::Class),
        ("interface_declaration", Kind::Interface),
        ("enum_declaration", Kind::Enum),
        ("type_alias_declaration", Kind::TypeAlias),
        ("method_definition", Kind::Method),
        ("arrow_function", Kind::Function),
        ("public_field_definition", Kind::Field),
        ("enum_assignment", Kind::Variant),
        ("variable_declarator", Kind::Const),
    ],
    container_kinds: &[Kind::Class, Kind::Enum, Kind::Interface, Kind::Module],
    call_nodes: &["call_expression"],
    call_field: "function",
    ref_node: "identifier",
    ref_field: "name",
    import_node: "import_statement",
    extra_skip_dirs: &[],
    transparent_scope_kinds: &[],
    extract_docs: true,
    instance_field_kinds: &[],
    no_retag_kinds: &[],
    impl_node: None,
    symbol_name_skip: &[],
    symbol_name_skip_prefix: &[],
};

pub static DART_CONFIG: LangConfig = LangConfig {
    scope_nodes: &[
        "function_signature",
        "function_body",
        "class_declaration",
        "mixin_declaration",
        "enum_declaration",
        "extension_declaration",
        "method_signature",
        "getter_signature",
        "setter_signature",
        "constructor_signature",
        "operator_signature",
        "factory_constructor_signature",
        "type_alias",
        "local_function_statement",
        "top_level_variable_declaration",
        "initialized_identifier",
        "static_final_declaration",
        "function_declaration",
        "method_declaration",
        "program",
    ],
    name_fields: &[
        NameField {
            node_kind: "function_signature",
            field: "name",
        },
        NameField {
            node_kind: "class_declaration",
            field: "name",
        },
        NameField {
            node_kind: "mixin_declaration",
            field: "name",
        },
        NameField {
            node_kind: "enum_declaration",
            field: "name",
        },
        NameField {
            node_kind: "extension_declaration",
            field: "name",
        },
        NameField {
            node_kind: "method_signature",
            field: "name",
        },
        NameField {
            node_kind: "getter_signature",
            field: "name",
        },
        NameField {
            node_kind: "setter_signature",
            field: "name",
        },
        NameField {
            node_kind: "constructor_signature",
            field: "name",
        },
        NameField {
            node_kind: "type_alias",
            field: "name",
        },
        NameField {
            node_kind: "enum_constant",
            field: "name",
        },
        NameField {
            node_kind: "field",
            field: "name",
        },
        NameField {
            node_kind: "static_final_declaration",
            field: "name",
        },
    ],
    kind_map: &[
        ("function_declaration", Kind::Function),
        ("method_declaration", Kind::Function),
        ("class_declaration", Kind::Class),
        ("mixin_declaration", Kind::Class),
        ("enum_declaration", Kind::Enum),
        ("enum_constant", Kind::Variant),
        ("extension_declaration", Kind::Class),
        ("getter_signature", Kind::Method),
        ("setter_signature", Kind::Method),
        ("constructor_signature", Kind::Method),
        ("operator_signature", Kind::Method),
        ("factory_constructor_signature", Kind::Method),
        ("type_alias", Kind::TypeAlias),
        ("field", Kind::Field),
        ("top_level_variable_declaration", Kind::Const),
        ("initialized_identifier", Kind::Const),
        ("static_final_declaration", Kind::Const),
    ],
    container_kinds: &[Kind::Class, Kind::Enum, Kind::Interface, Kind::Module],
    call_nodes: &["function_expression_invocation", "call_expression"],
    call_field: "name",
    ref_node: "identifier",
    ref_field: "name",
    import_node: "import_specification",
    extra_skip_dirs: &[],
    transparent_scope_kinds: &[],
    extract_docs: true,
    instance_field_kinds: &["initialized_identifier"],
    no_retag_kinds: &[],
    impl_node: None,
    // Flutter's `flutter gen-l10n` emits AppLocalizations (the base abstract
    // class) plus one AppLocalizationsXx per locale. The base name handles the
    // root file; the prefix sweep below handles every per-locale subclass.
    symbol_name_skip: &[],
    symbol_name_skip_prefix: &["AppLocalizations"],
};

pub static SOLIDITY_CONFIG: LangConfig = LangConfig {
    scope_nodes: &[
        "contract_declaration",
        "interface_declaration",
        "library_declaration",
        "function_definition",
        "modifier_definition",
        "event_definition",
        "error_declaration",
        "struct_declaration",
        "enum_declaration",
        "state_variable_declaration",
        "constant_variable_declaration",
        "using_directive",
        "source_unit",
    ],
    name_fields: &[
        NameField {
            node_kind: "contract_declaration",
            field: "name",
        },
        NameField {
            node_kind: "interface_declaration",
            field: "name",
        },
        NameField {
            node_kind: "library_declaration",
            field: "name",
        },
        NameField {
            node_kind: "function_definition",
            field: "name",
        },
        NameField {
            node_kind: "modifier_definition",
            field: "name",
        },
        NameField {
            node_kind: "event_definition",
            field: "name",
        },
        NameField {
            node_kind: "struct_declaration",
            field: "name",
        },
        NameField {
            node_kind: "enum_declaration",
            field: "name",
        },
        NameField {
            node_kind: "state_variable_declaration",
            field: "name",
        },
        NameField {
            node_kind: "constant_variable_declaration",
            field: "name",
        },
        NameField {
            node_kind: "enum_value",
            field: "name",
        },
        NameField {
            node_kind: "struct_member",
            field: "name",
        },
        NameField {
            node_kind: "user_defined_type_definition",
            field: "name",
        },
    ],
    kind_map: &[
        ("contract_declaration", Kind::Class),
        ("interface_declaration", Kind::Interface),
        ("library_declaration", Kind::Module),
        ("function_definition", Kind::Function),
        ("modifier_definition", Kind::Function),
        ("event_definition", Kind::Event),
        ("error_declaration", Kind::Error),
        ("struct_declaration", Kind::Struct),
        ("enum_declaration", Kind::Enum),
        ("enum_value", Kind::Variant),
        ("state_variable_declaration", Kind::Field),
        ("constant_variable_declaration", Kind::Const),
        ("struct_member", Kind::Field),
        ("constructor_definition", Kind::Method),
        ("fallback_receive_definition", Kind::Method),
        ("user_defined_type_definition", Kind::TypeAlias),
    ],
    container_kinds: &[
        Kind::Class,
        Kind::Interface,
        Kind::Module,
        Kind::Struct,
        Kind::Enum,
    ],
    call_nodes: &["function_call_expression", "call_expression"],
    call_field: "name",
    ref_node: "identifier",
    ref_field: "name",
    import_node: "import_directive",
    extra_skip_dirs: &["lib", "out", "cache", "broadcast"],
    transparent_scope_kinds: &[],
    extract_docs: false,
    instance_field_kinds: &[],
    no_retag_kinds: &["modifier_definition"],
    impl_node: None,
    symbol_name_skip: &[],
    symbol_name_skip_prefix: &[],
};
