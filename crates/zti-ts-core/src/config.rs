use tree_sitter::Node;

use crate::types::Kind;

#[derive(Debug, Clone)]
pub struct NameField {
    pub node_kind: &'static str,
    pub field: &'static str,
}

#[derive(Debug, Clone)]
pub struct LangConfig {
    pub scope_nodes: &'static [&'static str],
    pub name_fields: &'static [NameField],
    pub kind_map: &'static [(&'static str, Kind)],
    pub container_kinds: &'static [Kind],
    pub call_node: &'static str,
    pub call_field: &'static str,
    pub ref_node: &'static str,
    pub ref_field: &'static str,
    pub import_node: &'static str,
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

pub fn extract_name<'a>(node: &Node, source: &'a str, config: &LangConfig) -> Option<&'a str> {
    for field_def in config.name_fields {
        if node.kind() == field_def.node_kind
            && let Some(child) = node.child_by_field_name(field_def.field) {
                let text = child.utf8_text(source.as_bytes()).ok()?;
                if !text.is_empty() {
                    return Some(text);
                }
            }
    }
    None
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
        NameField { node_kind: "function_item", field: "name" },
        NameField { node_kind: "struct_item", field: "name" },
        NameField { node_kind: "enum_item", field: "name" },
        NameField { node_kind: "trait_item", field: "name" },
        NameField { node_kind: "mod_item", field: "name" },
        NameField { node_kind: "type_item", field: "name" },
        NameField { node_kind: "const_item", field: "name" },
        NameField { node_kind: "static_item", field: "name" },
        NameField { node_kind: "impl_item", field: "trait" },
        NameField { node_kind: "field_item", field: "name" },
        NameField { node_kind: "enum_variant", field: "name" },
        NameField { node_kind: "function_signature_item", field: "name" },
        NameField { node_kind: "metavariable", field: "name" },
        NameField { node_kind: "macro_invocation", field: "name" },
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
        ("impl_item", Kind::Module),
        ("field_item", Kind::Field),
        ("macro_definition", Kind::Function),
        ("metavariable", Kind::Const),
    ],
    container_kinds: &[Kind::Struct, Kind::Enum, Kind::Interface, Kind::Module, Kind::Class],
    call_node: "call_expression",
    call_field: "function",
    ref_node: "scoped_identifier",
    ref_field: "name",
    import_node: "use_declaration",
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
        NameField { node_kind: "function_declaration", field: "name" },
        NameField { node_kind: "class_declaration", field: "name" },
        NameField { node_kind: "interface_declaration", field: "name" },
        NameField { node_kind: "enum_declaration", field: "name" },
        NameField { node_kind: "type_alias_declaration", field: "name" },
        NameField { node_kind: "method_definition", field: "name" },
        NameField { node_kind: "public_field_definition", field: "name" },
        NameField { node_kind: "enum_assignment", field: "name" },
        NameField { node_kind: "property_identifier", field: "value" },
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
    ],
    container_kinds: &[Kind::Class, Kind::Enum, Kind::Interface, Kind::Module],
    call_node: "call_expression",
    call_field: "function",
    ref_node: "identifier",
    ref_field: "name",
    import_node: "import_statement",
};

pub static DART_CONFIG: LangConfig = LangConfig {
    scope_nodes: &[
        "function_signature",
        "function_body",
        "class_definition",
        "mixin_declaration",
        "enum_declaration",
        "extension_declaration",
        "method_signature",
        "getter_signature",
        "setter_signature",
        "constructor_signature",
        "type_alias",
        "local_function_statement",
        "program",
    ],
    name_fields: &[
        NameField { node_kind: "function_signature", field: "name" },
        NameField { node_kind: "class_definition", field: "name" },
        NameField { node_kind: "mixin_declaration", field: "name" },
        NameField { node_kind: "enum_declaration", field: "name" },
        NameField { node_kind: "extension_declaration", field: "name" },
        NameField { node_kind: "method_signature", field: "name" },
        NameField { node_kind: "getter_signature", field: "name" },
        NameField { node_kind: "setter_signature", field: "name" },
        NameField { node_kind: "constructor_signature", field: "name" },
        NameField { node_kind: "type_alias", field: "name" },
        NameField { node_kind: "enum_constant", field: "name" },
        NameField { node_kind: "field", field: "name" },
    ],
    kind_map: &[
        ("function_signature", Kind::Function),
        ("class_definition", Kind::Class),
        ("mixin_declaration", Kind::Class),
        ("enum_declaration", Kind::Enum),
        ("enum_constant", Kind::Variant),
        ("extension_declaration", Kind::Module),
        ("method_signature", Kind::Method),
        ("getter_signature", Kind::Method),
        ("setter_signature", Kind::Method),
        ("constructor_signature", Kind::Method),
        ("type_alias", Kind::TypeAlias),
        ("field", Kind::Field),
    ],
    container_kinds: &[Kind::Class, Kind::Enum, Kind::Module],
    call_node: "method_call_expression",
    call_field: "name",
    ref_node: "identifier",
    ref_field: "name",
    import_node: "import_specification",
};

pub static SOLIDITY_CONFIG: LangConfig = LangConfig {
    scope_nodes: &[
        "contract_declaration",
        "interface_declaration",
        "library_declaration",
        "function_definition",
        "modifier_definition",
        "event_definition",
        "error_definition",
        "struct_definition",
        "enum_definition",
        "state_variable_declaration",
        "using_directive",
        "source_unit",
    ],
    name_fields: &[
        NameField { node_kind: "contract_declaration", field: "name" },
        NameField { node_kind: "interface_declaration", field: "name" },
        NameField { node_kind: "library_declaration", field: "name" },
        NameField { node_kind: "function_definition", field: "name" },
        NameField { node_kind: "modifier_definition", field: "name" },
        NameField { node_kind: "event_definition", field: "name" },
        NameField { node_kind: "error_definition", field: "name" },
        NameField { node_kind: "struct_definition", field: "name" },
        NameField { node_kind: "enum_definition", field: "name" },
        NameField { node_kind: "state_variable_declaration", field: "name" },
        NameField { node_kind: "receive_function", field: "name" },
        NameField { node_kind: "fallback_function", field: "name" },
        NameField { node_kind: "constructor_definition", field: "name" },
        NameField { node_kind: "enum_value", field: "name" },
        NameField { node_kind: "struct_member", field: "name" },
    ],
    kind_map: &[
        ("contract_declaration", Kind::Class),
        ("interface_declaration", Kind::Interface),
        ("library_declaration", Kind::Module),
        ("function_definition", Kind::Function),
        ("modifier_definition", Kind::Function),
        ("event_definition", Kind::Event),
        ("error_definition", Kind::Error),
        ("struct_definition", Kind::Struct),
        ("enum_definition", Kind::Enum),
        ("enum_value", Kind::Variant),
        ("state_variable_declaration", Kind::Field),
        ("struct_member", Kind::Field),
        ("constructor_definition", Kind::Method),
        ("receive_function", Kind::Method),
        ("fallback_function", Kind::Method),
    ],
    container_kinds: &[Kind::Class, Kind::Interface, Kind::Module, Kind::Struct, Kind::Enum],
    call_node: "function_call_expression",
    call_field: "name",
    ref_node: "identifier",
    ref_field: "name",
    import_node: "import_directive",
};
