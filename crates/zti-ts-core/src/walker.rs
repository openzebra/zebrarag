use std::borrow::Cow;
use std::collections::HashMap;

use tree_sitter::{Node, Tree, TreeCursor};

use crate::config::LangConfig;
use crate::types::{Edge, EdgeKind, Kind, ParseResult, Symbol, Target};

fn squash_newlines(s: &str) -> Cow<'_, str> {
    if s.contains('\n') {
        Cow::Owned(s.replace('\n', " "))
    } else {
        Cow::Borrowed(s)
    }
}

pub fn parse_file(
    tree: &Tree,
    source: &str,
    file_idx: u16,
    config: &LangConfig,
    id_start: u32,
) -> (Vec<Symbol>, Vec<Edge>) {
    let mut symbols = Vec::with_capacity(64);
    let mut edges = Vec::with_capacity(64);
    let mut name_map: HashMap<String, u32> = HashMap::with_capacity(64);
    let mut state = WalkState {
        file_idx,
        next_id: id_start,
        scope_stack: Vec::with_capacity(8),
        qual_buf: String::with_capacity(128),
        qual_lens: Vec::with_capacity(8),
        edges: &mut edges,
        config,
        source,
        fn_depth: 0,
        container_depth: 0,
        scope_kinds: Vec::with_capacity(8),
    };
    let mut cursor = tree.root_node().walk();
    walk_node(&mut cursor, &mut symbols, &mut name_map, &mut state);
    (symbols, edges)
}

struct WalkState<'a> {
    file_idx: u16,
    next_id: u32,
    scope_stack: Vec<u32>,
    qual_buf: String,
    qual_lens: Vec<usize>,
    edges: &'a mut Vec<Edge>,
    config: &'a LangConfig,
    source: &'a str,
    fn_depth: u16,
    container_depth: u16,
    scope_kinds: Vec<Kind>,
}

impl<'a> WalkState<'a> {
    fn push_scope(&mut self, id: u32, name: &str, is_container: bool, is_fn: bool, kind: Kind) {
        self.scope_stack.push(id);
        let prev_len = self.qual_buf.len();
        if !self.qual_buf.is_empty() {
            self.qual_buf.push_str("::");
        }
        self.qual_buf.push_str(name);
        self.qual_lens.push(prev_len);
        self.scope_kinds.push(kind);
        if is_container {
            self.container_depth += 1;
        }
        if is_fn {
            self.fn_depth += 1;
        }
    }

    fn pop_scope(&mut self, was_container: bool, was_fn: bool) {
        self.scope_stack.pop();
        self.scope_kinds.pop();
        if let Some(prev_len) = self.qual_lens.pop() {
            self.qual_buf.truncate(prev_len);
        }
        if was_container {
            self.container_depth -= 1;
        }
        if was_fn {
            self.fn_depth -= 1;
        }
    }

    fn push_transparent(&mut self, id: u32) {
        self.scope_stack.push(id);
        self.qual_lens.push(self.qual_buf.len());
        self.scope_kinds.push(Kind::Module);
        self.container_depth += 1;
    }

    fn pop_transparent(&mut self) {
        self.scope_stack.pop();
        self.scope_kinds.pop();
        let _ = self.qual_lens.pop();
        self.container_depth -= 1;
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
        self.edges.push(Edge {
            from,
            to,
            kind,
            line,
        });
    }

    fn is_inside_fn(&self) -> bool {
        self.fn_depth > 0
    }

    fn is_inside_container(&self) -> bool {
        self.container_depth > 0
    }

    fn is_inside_method_container(&self) -> bool {
        self.scope_kinds.iter().any(|k| {
            matches!(
                k,
                Kind::Struct | Kind::Enum | Kind::Class | Kind::Impl | Kind::Interface
            )
        })
    }
}

fn walk_children(
    cursor: &mut TreeCursor,
    symbols: &mut Vec<Symbol>,
    name_map: &mut HashMap<String, u32>,
    state: &mut WalkState,
) {
    if cursor.goto_first_child() {
        loop {
            walk_node(cursor, symbols, name_map, state);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

fn walk_node(
    cursor: &mut TreeCursor,
    symbols: &mut Vec<Symbol>,
    name_map: &mut HashMap<String, u32>,
    state: &mut WalkState,
) {
    let node = cursor.node();

    for ts in state.config.transparent_scope_kinds {
        if node.kind() == ts.node_kind {
            let target_id = node
                .child_by_field_name(ts.target_field)
                .and_then(|c| c.utf8_text(state.source.as_bytes()).ok())
                .and_then(|name| name_map.get(name).copied());

            if let Some(id) = target_id {
                state.push_transparent(id);
                walk_children(cursor, symbols, name_map, state);
                state.pop_transparent();
            } else {
                walk_children(cursor, symbols, name_map, state);
            }
            return;
        }
    }

    if let Some(impl_node_kind) = state.config.impl_node
        && node.kind() == impl_node_kind
    {
        let type_name = node
            .child_by_field_name("type")
            .and_then(|c| c.utf8_text(state.source.as_bytes()).ok());

        let trait_name = node
            .child_by_field_name("trait")
            .and_then(|c| c.utf8_text(state.source.as_bytes()).ok());

        let Some(ty) = type_name else {
            walk_children(cursor, symbols, name_map, state);
            return;
        };

        let id = state.alloc_id();
        let parent = state.parent_id();

        let name = match trait_name {
            Some(t) => {
                let ty_clean = squash_newlines(ty);
                let t_clean = squash_newlines(t);
                let mut n = String::with_capacity(5 + t_clean.len() + 5 + ty_clean.len());
                n.push_str("impl ");
                n.push_str(&t_clean);
                n.push_str(" for ");
                n.push_str(&ty_clean);
                n
            }
            None => {
                let ty_clean = squash_newlines(ty);
                let mut n = String::with_capacity(5 + ty_clean.len());
                n.push_str("impl ");
                n.push_str(&ty_clean);
                n
            }
        };

        let qualified = if state.qual_buf.is_empty() {
            name.clone()
        } else {
            let mut q = String::with_capacity(state.qual_buf.len() + 2 + name.len());
            q.push_str(&state.qual_buf);
            q.push_str("::");
            q.push_str(&name);
            q
        };

        let (line, end_line) = lines_of(&node);
        let sig = collapse_signature(state.source, &node);

        let is_container = true;
        symbols.push(Symbol {
            id,
            kind: Kind::Impl,
            name,
            qualified,
            file_idx: state.file_idx,
            line,
            end_line,
            signature: sig,
            doc: None,
            base_classes: Vec::new(),
            parent,
            traits: Vec::new(),
        });

        state.push_scope(id, ty, is_container, false, Kind::Impl);
        collect_edges(&node, state, id);
        walk_children(cursor, symbols, name_map, state);
        state.pop_scope(is_container, false);
        return;
    }

    if let Some(mut kind) = state.config.kind_for(&node) {
        if kind == Kind::Const && state.is_inside_fn() {
            walk_children(cursor, symbols, name_map, state);
            return;
        }

        if should_skip_paramless_function_inside_fn(&node, state, kind) {
            walk_children(cursor, symbols, name_map, state);
            return;
        }

        if state.config.instance_field_kinds.contains(&node.kind())
            && !state.is_inside_fn()
            && state.is_inside_container()
            && kind == Kind::Const
        {
            walk_children(cursor, symbols, name_map, state);
            return;
        }

        if kind == Kind::Const && node.kind() == "variable_declarator" {
            let has_arrow_value = node
                .child_by_field_name("value")
                .is_some_and(|v| v.kind() == "arrow_function");
            if has_arrow_value {
                kind = Kind::Function;
            }
        }

        if kind == Kind::Function
            && state.is_inside_method_container()
            && !state.config.no_retag_kinds.contains(&node.kind())
        {
            kind = Kind::Method;
        }

        let Some(name) = crate::config::extract_name(&node, state.source, state.config) else {
            walk_children(cursor, symbols, name_map, state);
            return;
        };

        if state.config.symbol_name_skip.contains(&name.as_str())
            || state
                .config
                .symbol_name_skip_prefix
                .iter()
                .any(|p| name.starts_with(p))
        {
            return;
        }

        let id = state.alloc_id();
        let parent = state.parent_id();

        let qualified = if state.qual_buf.is_empty() {
            name.clone()
        } else {
            let mut q = String::with_capacity(state.qual_buf.len() + 2 + name.len());
            q.push_str(&state.qual_buf);
            q.push_str("::");
            q.push_str(&name);
            q
        };

        let (line, end_line) = lines_of(&node);

        let sig = collapse_signature(state.source, &node);

        let doc = extract_doc(&node, state.source, state.config);

        let base_classes = extract_base_classes(&node, state.source);
        let traits = extract_traits(&node, state.source);

        let is_container = state.config.container_kinds.contains(&kind);
        let is_fn = kind == Kind::Function || kind == Kind::Method;

        if is_container {
            name_map.insert(name.clone(), id);
        }

        symbols.push(Symbol {
            id,
            kind,
            name,
            qualified,
            file_idx: state.file_idx,
            line,
            end_line,
            signature: sig,
            doc,
            base_classes,
            parent,
            traits,
        });

        let name = symbols
            .last()
            .map(|s| s.name.as_str())
            .unwrap_or("<unknown>");
        state.push_scope(id, name, is_container, is_fn, kind);

        collect_edges(&node, state, id);

        walk_children(cursor, symbols, name_map, state);

        state.pop_scope(is_container, is_fn);
        return;
    }

    walk_children(cursor, symbols, name_map, state);
}

fn should_skip_paramless_function_inside_fn(node: &Node, state: &WalkState, kind: Kind) -> bool {
    state.config.skip_paramless_functions_inside_fn
        && kind == Kind::Function
        && state.is_inside_fn()
        && !has_child_kind(node, "parameter")
}

fn has_child_kind(node: &Node, child_kind: &str) -> bool {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .any(|child| child.kind() == child_kind)
}

fn lines_of(node: &Node) -> (u32, u32) {
    let start = node.start_position().row as u32 + 1;
    let end = node.end_position().row as u32 + 1;
    (start, end)
}

fn collapse_signature(source: &str, node: &Node) -> String {
    let span = node.utf8_text(source.as_bytes()).unwrap_or("");
    let end = span.find(['{', ';']).unwrap_or(span.len());
    let mut sig = String::with_capacity(end.min(256));
    let mut last_was_space = true;
    for ch in span[..end].chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                sig.push(' ');
                last_was_space = true;
            }
        } else {
            sig.push(ch);
            last_was_space = false;
        }
    }
    sig.trim_end().to_string()
}

fn is_doc_marker(text: &str) -> bool {
    text.starts_with("///")
        || text.starts_with("//!")
        || text.starts_with("/**")
        || text.starts_with("/*!")
}

fn strip_doc_markers(text: &str) -> &str {
    let trimmed = if let Some(rest) = text.strip_prefix("///") {
        rest
    } else if let Some(rest) = text.strip_prefix("//!") {
        rest
    } else if let Some(rest) = text.strip_prefix("/**") {
        rest.trim_end_matches("*/")
    } else if let Some(rest) = text.strip_prefix("/*!") {
        rest.trim_end_matches("*/")
    } else {
        text
    };
    trimmed.trim()
}

fn extract_doc(node: &Node, source: &str, config: &LangConfig) -> Option<String> {
    if !config.extract_docs {
        return None;
    }

    let mut prev = node.prev_named_sibling()?;
    while matches!(prev.kind(), "attribute_item" | "attribute" | "meta") {
        prev = prev.prev_named_sibling()?;
    }

    // Walk backwards collecting contiguous doc-comment siblings (newest first).
    let mut lines: Vec<&str> = Vec::with_capacity(4);
    let mut cur = Some(prev);
    while let Some(n) = cur {
        let text = n.utf8_text(source.as_bytes()).ok()?;
        if !is_doc_marker(text) {
            break;
        }
        lines.push(text);
        cur = n.prev_named_sibling();
    }
    if lines.is_empty() {
        return None;
    }
    lines.reverse();

    let total: usize = lines.iter().map(|s| s.len()).sum::<usize>() + lines.len();
    let mut buf = String::with_capacity(total);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            buf.push('\n');
        }
        buf.push_str(strip_doc_markers(line));
    }
    Some(buf)
}

fn extract_base_classes(node: &Node, source: &str) -> Vec<String> {
    let mut bases = Vec::with_capacity(4);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if matches!(
            child.kind(),
            "superclass" | "base_class_clause" | "implements_clause"
        ) {
            collect_identifiers(&child, source, &mut bases);
        }
    }
    bases
}

fn extract_traits(node: &Node, source: &str) -> Vec<String> {
    let mut traits = Vec::with_capacity(4);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if matches!(
            child.kind(),
            "trait_bounds" | "implements_clause" | "with_clause"
        ) {
            collect_identifiers(&child, source, &mut traits);
        }
    }
    traits
}

fn collect_identifiers(node: &Node, source: &str, out: &mut Vec<String>) {
    if (node.kind() == "identifier"
        || node.kind() == "type_identifier"
        || node.kind() == "scoped_identifier")
        && let Ok(text) = node.utf8_text(source.as_bytes())
    {
        out.push(text.to_string());
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_identifiers(&child, source, out);
    }
}

fn collect_edges(node: &Node, state: &mut WalkState, from_id: u32) {
    let call_field = state.config.call_field;
    let ref_kind = state.config.ref_node;

    collect_edges_recursive(node, state, from_id, call_field, ref_kind);
}

fn collect_edges_recursive(
    node: &Node,
    state: &mut WalkState,
    from_id: u32,
    call_field: &'static str,
    ref_kind: &'static str,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        // Stop at nested symbol boundaries — their own emit collects their edges.
        // Without this, calls inside nested fns/methods get attributed to the
        // outer scope too (double-counting).
        if state.config.kind_for(&child).is_some() {
            continue;
        }

        let kind = child.kind();

        if state.config.call_nodes.contains(&kind) {
            let func_node = child
                .child_by_field_name(call_field)
                .or_else(|| child.child(0));
            if let Some(fn_node) = func_node {
                let callee = resolve_call_name(&fn_node, state.source);
                let line = child.start_position().row as u32 + 1;
                if !callee.is_empty() {
                    state.add_edge(from_id, Target::Unresolved(callee), EdgeKind::Call, line);
                }
            }
        }

        if kind == ref_kind {
            let text = child.utf8_text(state.source.as_bytes()).unwrap_or("");
            if !text.is_empty() {
                let line = child.start_position().row as u32 + 1;
                state.add_edge(
                    from_id,
                    Target::Unresolved(text.to_string()),
                    EdgeKind::Ref,
                    line,
                );
            }
        }

        if child.child_count() > 0 {
            collect_edges_recursive(&child, state, from_id, call_field, ref_kind);
        }
    }
}

fn resolve_call_name(node: &Node, source: &str) -> String {
    if node.kind() == "scoped_identifier"
        || node.kind() == "field_expression"
        || node.kind() == "member_expression"
        || node.kind() == "selector_expression"
    {
        return node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
    }
    if node.kind() == "identifier" || node.kind() == "property_identifier" {
        return node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
    }
    if (node.kind() == "generic_function" || node.kind() == "generic_type")
        && let Some(child) = node.child(0)
    {
        return resolve_call_name(&child, source);
    }
    node.utf8_text(source.as_bytes()).unwrap_or("").to_string()
}

pub trait LanguageFrontend {
    fn language(&self) -> tree_sitter::Language;
    fn config(&self) -> &'static LangConfig;
    fn extract_imports(&self, root: Node, source: &str) -> HashMap<String, String>;

    fn parse(&self, source: &str, file_idx: u16, id_start: u32) -> ParseResult {
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&self.language())?;
        let tree = parser
            .parse(source, None)
            .ok_or_else(|| anyhow::anyhow!("parse failed"))?;

        let (symbols, edges) = parse_file(&tree, source, file_idx, self.config(), id_start);
        let imports = self.extract_imports(tree.root_node(), source);

        Ok((symbols, edges, imports))
    }
}
