use std::collections::HashMap;

use anyhow::Result;
use tree_sitter::{Node, Tree, TreeCursor};

use crate::config::{extract_name, LangConfig};
use crate::types::{Edge, EdgeKind, Symbol, Target};

pub fn parse_file(
    tree: &Tree,
    source: &str,
    file_idx: u16,
    config: &LangConfig,
    id_start: u32,
) -> (Vec<Symbol>, Vec<Edge>) {
    let mut symbols = Vec::new();
    let mut edges = Vec::new();
    let mut state = WalkState {
        file_idx,
        next_id: id_start,
        scope_stack: Vec::new(),
        scope_qual: Vec::new(),
        edges: &mut edges,
        config,
        source,
    };
    let mut cursor = tree.root_node().walk();
    walk_node(&mut cursor, &mut symbols, &mut state);
    (symbols, edges)
}

struct WalkState<'a> {
    file_idx: u16,
    next_id: u32,
    scope_stack: Vec<u32>,
    scope_qual: Vec<String>,
    edges: &'a mut Vec<Edge>,
    config: &'a LangConfig,
    source: &'a str,
}

impl<'a> WalkState<'a> {
    fn push_scope(&mut self, id: u32, qual: String) {
        self.scope_stack.push(id);
        self.scope_qual.push(qual);
    }

    fn pop_scope(&mut self) {
        self.scope_stack.pop();
        self.scope_qual.pop();
    }

    fn current_qual(&self) -> String {
        self.scope_qual.join("::")
    }

    fn parent_id(&self) -> Option<u32> {
        self.scope_stack.last().copied()
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn add_edge(&mut self, from: u32, to: Target, kind: EdgeKind, line: u32) {
        self.edges.push(Edge { from, to, kind, line });
    }
}

fn walk_node(cursor: &mut TreeCursor, symbols: &mut Vec<Symbol>, state: &mut WalkState) {
    let node = cursor.node();

    if let Some(kind) = state.config.kind_for(&node) {
        let name = extract_name(&node, state.source, state.config)
            .unwrap_or("")
            .to_string();

        let id = state.alloc_id();
        let parent = state.parent_id();

        let qual_prefix = state.current_qual();
        let qualified = if qual_prefix.is_empty() {
            name.clone()
        } else {
            format!("{}::{}", qual_prefix, name)
        };

        let (line, end_line) = lines_of(&node);

        let sig = node.utf8_text(state.source.as_bytes())
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("")
            .to_string();

        let doc = extract_doc(&node, state.source);

        let base_classes = extract_base_classes(&node, state.source, state.config);
        let traits = extract_traits(&node, state.source, state.config);

        symbols.push(Symbol {
            id,
            kind,
            name: name.clone(),
            qualified: qualified.clone(),
            file_idx: state.file_idx,
            line,
            end_line,
            signature: sig,
            doc,
            base_classes,
            parent,
            traits,
        });

        state.push_scope(id, name);

        collect_edges(&node, state, id);

        if cursor.goto_first_child() {
            loop {
                walk_node(cursor, symbols, state);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }

        state.pop_scope();
        return;
    }

    if cursor.goto_first_child() {
        loop {
            walk_node(cursor, symbols, state);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

fn lines_of(node: &Node) -> (u32, u32) {
    let start = node.start_position().row + 1;
    let end = node.end_position().row;
    (start as u32, end as u32)
}

fn extract_doc(node: &Node, source: &str) -> Option<String> {
    let prev = node.prev_named_sibling()?;
    let text = prev.utf8_text(source.as_bytes()).ok()?;
    if text.starts_with("///") || text.starts_with("/**") || text.starts_with("//") {
        Some(text.to_string())
    } else {
        None
    }
}

fn extract_base_classes(node: &Node, source: &str, _config: &LangConfig) -> Vec<String> {
    let mut bases = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "superclass" || child.kind() == "base_class_clause" || child.kind() == "implements_clause" {
            collect_identifiers(&child, source, &mut bases);
        }
        if child.kind() == "type_arguments" || child.kind() == "generic_type" {
            collect_identifiers(&child, source, &mut bases);
        }
    }
    bases
}

fn extract_traits(node: &Node, source: &str, _config: &LangConfig) -> Vec<String> {
    let mut traits = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "trait_bounds" || child.kind() == "implements_clause" || child.kind() == "with_clause" {
            collect_identifiers(&child, source, &mut traits);
        }
    }
    traits
}

fn collect_identifiers(node: &Node, source: &str, out: &mut Vec<String>) {
    if (node.kind() == "identifier" || node.kind() == "type_identifier" || node.kind() == "scoped_identifier")
        && let Ok(text) = node.utf8_text(source.as_bytes()) {
            out.push(text.to_string());
        }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_identifiers(&child, source, out);
    }
}

fn collect_edges(node: &Node, state: &mut WalkState, from_id: u32) {
    collect_call_edges(node, state, from_id);
    collect_ref_edges(node, state, from_id);
}

fn collect_call_edges(node: &Node, state: &mut WalkState, from_id: u32) {
    let call_kind = state.config.call_node;
    let call_field = state.config.call_field;

    for child in node.children(&mut node.walk()) {
        if child.kind() == call_kind
            && let Some(func_node) = child.child_by_field_name(call_field) {
                let callee = resolve_call_name(&func_node, state.source);
                let line = child.start_position().row as u32 + 1;
                state.add_edge(from_id, Target::Unresolved(callee), EdgeKind::Call, line);
            }
        if child.kind() != call_kind && child.child_count() > 0 {
            collect_call_edges(&child, state, from_id);
        }
    }
}

fn resolve_call_name(node: &Node, source: &str) -> String {
    if node.kind() == "scoped_identifier" || node.kind() == "field_expression" || node.kind() == "member_expression" {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("");
        return text.to_string();
    }
    if node.kind() == "selector_expression" {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("");
        return text.to_string();
    }
    if node.kind() == "identifier" || node.kind() == "property_identifier" {
        return node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
    }
    if (node.kind() == "generic_function" || node.kind() == "generic_type")
        && let Some(child) = node.child(0) {
            return resolve_call_name(&child, source);
        }
    node.utf8_text(source.as_bytes()).unwrap_or("").to_string()
}

fn collect_ref_edges(node: &Node, state: &mut WalkState, from_id: u32) {
    let ref_kind = state.config.ref_node;

    for child in node.children(&mut node.walk()) {
        if child.kind() == ref_kind {
            let text = child.utf8_text(state.source.as_bytes()).unwrap_or("").to_string();
            let line = child.start_position().row as u32 + 1;
            if !text.is_empty() {
                state.add_edge(from_id, Target::Unresolved(text), EdgeKind::Ref, line);
            }
        }
    }
}

pub trait LanguageFrontend {
    fn language(&self) -> tree_sitter::Language;
    fn config(&self) -> &'static LangConfig;
    fn parse(&self, source: &str, file_idx: u16, id_start: u32) -> Result<(Vec<Symbol>, Vec<Edge>, HashMap<String, String>)>;
}

pub fn extract_imports_generic(node: &Node, source: &str, import_node_kind: &str) -> HashMap<String, String> {
    let mut imports = HashMap::new();
    do_extract_imports(node, source, import_node_kind, &mut imports);
    imports
}

fn do_extract_imports(node: &Node, source: &str, import_kind: &str, imports: &mut HashMap<String, String>) {
    if node.kind() == import_kind {
        let text = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
        let local_name = extract_import_local_name(node, source);
        if let Some(name) = local_name {
            imports.entry(name).or_insert(text);
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        do_extract_imports(&child, source, import_kind, imports);
    }
}

fn extract_import_local_name(node: &Node, source: &str) -> Option<String> {
    for child in node.children(&mut node.walk()) {
        let kind = child.kind();
        if kind == "identifier" || kind == "property_identifier" {
            return child.utf8_text(source.as_bytes()).ok().map(|s| s.to_string());
        }
    }
    None
}
