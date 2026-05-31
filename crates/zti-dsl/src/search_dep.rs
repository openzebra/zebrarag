use std::fmt::Write as _;

use zti_common::dsl::SymbolBodyEntry;
use zti_ts_core::types::Symbol;

use crate::batch::resolve_symbol_bodies;
use crate::model::ProjectIndex;
use crate::AsciiTreeRenderer;

const BYTES_PER_TOKEN: usize = 4;

#[derive(Debug)]
pub enum NameMatch {
    Found(u32),
    Ambiguous(Vec<u32>),
    NotFound,
}

pub fn resolve_name(index: &ProjectIndex, name: &str) -> NameMatch {
    if let Some(&id) = index.qualified_map.get(name) {
        return NameMatch::Found(id);
    }
    let mut ids = Vec::with_capacity(4);
    for s in &index.symbols {
        if s.name == name || s.qualified == name {
            ids.push(s.id);
        }
    }
    match ids.len() {
        0 => NameMatch::NotFound,
        1 => NameMatch::Found(ids[0]),
        _ => NameMatch::Ambiguous(ids),
    }
}

fn find_symbol(index: &ProjectIndex, id: u32) -> Option<&Symbol> {
    index.symbols.get(id as usize)
}

fn trim_tree_header(s: &str) -> &str {
    if let Some(pos) = s.find('\n') {
        s[pos + 1..].trim_end()
    } else {
        ""
    }
}

pub fn render_symbol_overview(
    index: &ProjectIndex,
    id: u32,
    depth: usize,
    max_tokens: usize,
) -> String {
    let budget = max_tokens.saturating_mul(BYTES_PER_TOKEN);
    let mut out = String::with_capacity(budget.min(8192));

    let Some(sym) = find_symbol(index, id) else {
        let _ = write!(out, "Symbol #{id} not found");
        return out;
    };

    // Header: file path only
    let file = index
        .files
        .get(sym.file_idx as usize)
        .map(|f| f.path.as_str())
        .unwrap_or("?");
    let _ = writeln!(out, "{file}:{}-{}", sym.line, sym.end_line);

    if let Some(ref doc) = sym.doc
        && let Some(first) = doc.lines().next()
    {
        let trimmed = first.trim();
        if !trimmed.is_empty() {
            out.push('\n');
            let _ = writeln!(out, "{trimmed}");
        }
    }

    // Call chains: entry points → target
    if depth > 0 {
        let renderer = AsciiTreeRenderer::new(index);
        let chains = renderer.render_call_chains(id, depth);
        if !chains.is_empty() {
            out.push('\n');
            out.push_str(&chains);
        }

        // Callees: project-internal only
        let callees = renderer.render_callees_clean(id, depth);
        let body = trim_tree_header(&callees);
        if !body.is_empty() {
            let _ = writeln!(out, "-- callees:");
            let _ = writeln!(out, "{body}");
        }
    }

    let bodies = resolve_symbol_bodies(index, &[id]);
    if let Some(SymbolBodyEntry::Ok { body, .. }) = bodies.into_iter().next() {
        out.push('\n');
        let _ = write!(out, "{body}");
    }

    truncate_budget(&mut out, budget);
    out
}

pub fn render_candidates(index: &ProjectIndex, ids: &[u32]) -> String {
    let mut out = String::with_capacity(32 + ids.len() * 80);
    out.push_str("Ambiguous name — call searchDep again with a qualified path:\n");
    for &id in ids {
        if let Some(s) = find_symbol(index, id) {
            let file = index
                .files
                .get(s.file_idx as usize)
                .map(|f| f.path.as_str())
                .unwrap_or("?");
            let _ = writeln!(
                out,
                "  #{id} : {} {} ({file}:{}-{})",
                s.kind.as_str(),
                s.qualified,
                s.line,
                s.end_line,
            );
        }
    }
    out
}

pub fn truncate_budget(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut cut = max_bytes.saturating_sub(1);
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s.truncate(cut);
    s.push_str("\n… [truncated: token budget reached]\n");
}
