use std::fmt::Write as _;

use zti_common::dsl::SymbolBodyEntry;
use zti_ts_core::types::Symbol;

use crate::batch::resolve_symbol_bodies;
use crate::model::ProjectIndex;
use crate::AsciiTreeRenderer;

const BYTES_PER_TOKEN: usize = 4;
const MAX_CANDIDATE_ALIASES: usize = 3;

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

fn is_name_duplicated(index: &ProjectIndex, name: &str) -> bool {
    index
        .symbols
        .iter()
        .filter(|sym| sym.name == name)
        .take(2)
        .count()
        > 1
}

fn alias_rank(sym: &Symbol, alias: &str) -> u8 {
    if alias == sym.qualified && alias.contains("::") {
        return 0;
    }

    let ends_with_name = alias
        .rsplit_once("::")
        .is_some_and(|(_, last)| last == sym.name);
    if ends_with_name && alias.matches("::").count() == 1 {
        return 1;
    }

    if ends_with_name && !alias.starts_with("crates::") && !alias.contains('-') {
        return 2;
    }

    3
}

fn candidate_aliases<'a>(index: &'a ProjectIndex, sym: &'a Symbol) -> Vec<&'a str> {
    let is_duplicated = is_name_duplicated(index, sym.name.as_str());
    let mut aliases = Vec::with_capacity(MAX_CANDIDATE_ALIASES.saturating_mul(2));

    if sym.qualified.contains("::")
        && matches!(resolve_name(index, sym.qualified.as_str()), NameMatch::Found(id) if id == sym.id)
    {
        aliases.push(sym.qualified.as_str());
    }

    for (alias, id) in &index.qualified_map {
        let alias = alias.as_str();
        if *id == sym.id
            && (!is_duplicated || alias.contains("::"))
            && aliases.iter().all(|existing| existing != &alias)
        {
            aliases.push(alias);
        }
    }

    aliases.sort_by(|left, right| {
        alias_rank(sym, left)
            .cmp(&alias_rank(sym, right))
            .then_with(|| left.cmp(right))
    });
    aliases.truncate(MAX_CANDIDATE_ALIASES);
    aliases
}

pub fn render_candidates(index: &ProjectIndex, ids: &[u32]) -> String {
    let mut out = String::with_capacity(64 + ids.len() * 112);
    out.push_str("Ambiguous name — call searchDep again with one of these exact names:\n");
    for &id in ids {
        if let Some(s) = find_symbol(index, id) {
            let file = index
                .files
                .get(s.file_idx as usize)
                .map(|f| f.path.as_str())
                .unwrap_or("?");
            let aliases = candidate_aliases(index, s);
            let display_name = aliases.first().copied().unwrap_or(s.qualified.as_str());
            let _ = write!(out, "  #{id} : {} {}", s.kind.as_str(), display_name);
            let mut extra_aliases = aliases.iter().skip(1);
            if let Some(first_alias) = extra_aliases.next() {
                let _ = write!(out, " (aliases: {first_alias}");
                for alias in extra_aliases {
                    let _ = write!(out, ", {alias}");
                }
                out.push(')');
            }
            let _ = writeln!(out, " ({file}:{}-{})", s.line, s.end_line);
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
