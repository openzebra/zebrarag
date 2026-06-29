use std::collections::HashSet;
use std::fmt::Write as _;

use zrag_ts_core::types::{Edge, EdgeKind, Target};

use crate::model::ProjectIndex;

pub struct AsciiTreeRenderer<'a> {
    index: &'a ProjectIndex,
}

impl<'a> AsciiTreeRenderer<'a> {
    pub fn new(index: &'a ProjectIndex) -> Self {
        Self { index }
    }

    /// Render call chains from entry points down to the target symbol.
    /// Entry points are symbols with outgoing call edges but no incoming
    /// project-internal (Resolved) call edges.
    pub fn render_call_chains(&self, target_id: u32, max_depth: usize) -> String {
        let mut out = String::with_capacity(1024);

        // Build set of symbols that receive at least one project-internal call edge.
        let mut has_incoming: HashSet<u32> = HashSet::with_capacity(self.index.symbols.len() / 4);
        for edges in self.index.reverse_edges.values() {
            for e in edges {
                if e.kind == EdgeKind::Call
                    && let Target::Resolved(id) = e.to
                {
                    has_incoming.insert(id);
                }
            }
        }

        // Collect all paths from entry points (no incoming calls) to target.
        let mut paths: Vec<Vec<u32>> = Vec::with_capacity(8);
        let mut current: Vec<u32> = Vec::with_capacity(16);
        let mut visited: HashSet<u32> = HashSet::with_capacity(64);
        self.collect_paths_to(
            target_id,
            max_depth,
            &has_incoming,
            &mut current,
            &mut visited,
            &mut paths,
        );

        if paths.is_empty() {
            return out;
        }

        // Sort paths: shorter first, then by first symbol name for stability.
        paths.sort_by(|a, b| {
            a.len().cmp(&b.len()).then_with(|| {
                self.sym_name(a.first().copied())
                    .cmp(self.sym_name(b.first().copied()))
            })
        });

        out.push_str("Call chains:\n");
        out.push_str("Entry points:\n");

        let total = paths.len();
        let mut chain_prefix = String::with_capacity(max_depth * 4);
        for (i, path) in paths.iter().enumerate() {
            let is_last = i + 1 == total;
            chain_prefix.clear();
            self.render_one_chain(&mut out, path, is_last, &mut chain_prefix);
        }

        out
    }

    /// DFS backwards from `id` through reverse call edges, collecting paths
    /// that end at entry points (symbols with no incoming calls).
    fn collect_paths_to(
        &self,
        id: u32,
        max_depth: usize,
        has_incoming: &HashSet<u32>,
        current: &mut Vec<u32>,
        visited: &mut HashSet<u32>,
        paths: &mut Vec<Vec<u32>>,
    ) {
        if current.len() >= max_depth || !visited.insert(id) {
            return;
        }
        current.push(id);

        let edges = self
            .index
            .reverse_edges
            .get(&id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        let mut callers = Vec::with_capacity(edges.len());
        for e in edges {
            if e.kind == EdgeKind::Call
                && let Target::Resolved(from_id) = e.to
            {
                callers.push(from_id);
            }
        }

        if callers.is_empty() && !has_incoming.contains(&id) {
            // Entry point reached: record path (current is from target → entry).
            // Reverse so it reads entry → ... → target.
            let mut rev: Vec<u32> = Vec::with_capacity(current.len());
            rev.extend(current.iter().rev().copied());
            paths.push(rev);
        } else {
            for caller_id in callers {
                self.collect_paths_to(caller_id, max_depth, has_incoming, current, visited, paths);
            }
        }

        current.pop();
        visited.remove(&id);
    }

    fn sym_name(&self, id: Option<u32>) -> &str {
        id.and_then(|i| self.index.symbols.get(i as usize))
            .map(|s| s.name.as_str())
            .unwrap_or("")
    }

    fn sym_file_line(&self, id: u32) -> (u32, u32) {
        self.index
            .symbols
            .get(id as usize)
            .map(|s| (s.line, s.end_line))
            .unwrap_or((0, 0))
    }

    #[allow(clippy::only_used_in_recursion)]
    fn render_one_chain(&self, out: &mut String, path: &[u32], is_last: bool, prefix: &mut String) {
        if path.is_empty() {
            return;
        }
        let branch = if is_last { "└─ " } else { "├─ " };
        let child_segment = if is_last { "   " } else { "│  " };

        let id = path[0];
        let name = self.sym_name(Some(id));
        let (line, _) = self.sym_file_line(id);
        let file = self
            .index
            .files
            .get(
                self.index
                    .symbols
                    .get(id as usize)
                    .map(|s| s.file_idx as usize)
                    .unwrap_or(0),
            )
            .map(|f| {
                f.path
                    .strip_prefix(&self.index.root)
                    .unwrap_or(&f.path)
                    .trim_start_matches('/')
            })
            .unwrap_or("?");

        out.push_str(prefix);
        out.push_str(branch);
        let _ = write!(out, "{name} ({file}:{line})");
        out.push('\n');

        let rest = &path[1..];
        if !rest.is_empty() {
            let saved = prefix.len();
            prefix.push_str(child_segment);
            self.render_one_chain(out, rest, true, prefix);
            prefix.truncate(saved);
        }
    }

    pub fn render_callers(&self, id: u32, max_depth: usize) -> String {
        let mut out = String::with_capacity(512);
        let sym = match self.index.symbols.get(id as usize) {
            Some(s) => s,
            None => return format!("Symbol {} not found\n", id),
        };
        let _ = writeln!(
            out,
            "{}#{} {} (callers)",
            sym.kind.short(),
            id,
            sym.qualified
        );
        let mut prefix = String::with_capacity(max_depth * 4);
        let mut visited = HashSet::with_capacity(64);
        self.recurse(
            id,
            max_depth,
            0,
            &mut out,
            &mut prefix,
            &mut visited,
            Direction::Callers,
            false,
            true,
        );
        out
    }

    pub fn render_callees(&self, id: u32, max_depth: usize, local_only: bool) -> String {
        self.render_callees_with_ids(id, max_depth, local_only, true)
    }

    /// Build a human-readable qualified display name for a symbol.
    /// Uses the walker-qualified name if it has parent scope; otherwise falls
    /// back to file-basename qualification (the same scheme used for
    /// search-dep disambiguation).
    fn sym_display(&self, id: u32) -> String {
        let sym = match self.index.symbols.get(id as usize) {
            Some(s) => s,
            None => return format!("#{id}"),
        };
        // Already qualified (e.g. "StorageLayout::BookLevel") — use as-is.
        if sym.qualified.contains("::") {
            return sym.qualified.clone();
        }
        // Build file-basename qualification.
        if let Some(file) = self.index.files.get(sym.file_idx as usize) {
            let short = file
                .path
                .rsplit('/')
                .next()
                .unwrap_or(&file.path)
                .trim_end_matches(".rs")
                .trim_end_matches(".ts")
                .trim_end_matches(".tsx")
                .trim_end_matches(".dart")
                .trim_end_matches(".sol");
            // Non-module basenames (mod.rs, lib.rs, main.rs, index.ts) — use
            // the parent directory name instead.
            let qual = if matches!(short, "mod" | "lib" | "main" | "index") {
                file.path.rsplit('/').nth(1).unwrap_or(short)
            } else {
                short
            };
            if qual != sym.name {
                return format!("{}::{}", qual, sym.qualified);
            }
        }
        sym.qualified.clone()
    }

    pub fn render_callees_clean(&self, id: u32, max_depth: usize) -> String {
        let mut out = String::with_capacity(512);
        let display = self.sym_display(id);
        let _ = writeln!(out, "{} (callees)", display);
        let mut prefix = String::with_capacity(max_depth * 4);
        let mut visited = HashSet::with_capacity(64);
        self.recurse_clean(
            id,
            max_depth,
            0,
            &mut out,
            &mut prefix,
            &mut visited,
            Direction::Callees,
        );
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn recurse_clean(
        &self,
        id: u32,
        max_depth: usize,
        depth: usize,
        out: &mut String,
        prefix: &mut String,
        visited: &mut HashSet<u32>,
        direction: Direction,
    ) {
        if depth >= max_depth || !visited.insert(id) {
            return;
        }

        let edges_for_id: &[Edge] = match direction {
            Direction::Callers => self
                .index
                .reverse_edges
                .get(&id)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            Direction::Callees => self
                .index
                .forward_edges
                .get(&id)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        };

        let mut seen_external = HashSet::with_capacity(edges_for_id.len());
        let filtered: Vec<&Edge> = edges_for_id
            .iter()
            .filter(|e| e.kind == EdgeKind::Call)
            .filter(|e| match &e.to {
                Target::External(name) => seen_external.insert(name.as_str()),
                Target::Resolved(_) | Target::Unresolved(_) => true,
            })
            .collect();

        let total = filtered.len();
        if total == 0 {
            return;
        }

        for (i, edge) in filtered.iter().enumerate() {
            let is_last = i + 1 == total;
            let branch = if is_last { "└── " } else { "├── " };
            let child_segment = if is_last { "    " } else { "│   " };

            out.push_str(prefix);
            out.push_str(branch);

            match &edge.to {
                Target::Resolved(to_id) => {
                    if let Some(sym) = self.index.symbols.get(*to_id as usize) {
                        let display = self.sym_display(*to_id);
                        let file = self
                            .index
                            .files
                            .get(sym.file_idx as usize)
                            .map(|f| {
                                f.path
                                    .strip_prefix(&self.index.root)
                                    .unwrap_or(&f.path)
                                    .trim_start_matches('/')
                            })
                            .unwrap_or("?");
                        let _ = writeln!(out, "{} ({}:{})", display, file, sym.line);
                        let saved = prefix.len();
                        prefix.push_str(child_segment);
                        self.recurse_clean(
                            *to_id,
                            max_depth,
                            depth + 1,
                            out,
                            prefix,
                            visited,
                            direction,
                        );
                        prefix.truncate(saved);
                    } else {
                        out.push('\n');
                    }
                }
                Target::External(name) => {
                    out.push_str(name);
                    out.push('\n');
                }
                Target::Unresolved(_) => out.push('\n'),
            }
        }
    }

    pub fn render_callees_with_ids(
        &self,
        id: u32,
        max_depth: usize,
        local_only: bool,
        show_ids: bool,
    ) -> String {
        let mut out = String::with_capacity(512);
        let sym = match self.index.symbols.get(id as usize) {
            Some(s) => s,
            None => return format!("Symbol {} not found\n", id),
        };
        if show_ids {
            let _ = writeln!(
                out,
                "{}#{} {} (callees)",
                sym.kind.short(),
                id,
                sym.qualified
            );
        } else {
            let _ = writeln!(out, "{} {} (callees)", sym.kind.short(), sym.qualified);
        }
        let mut prefix = String::with_capacity(max_depth * 4);
        let mut visited = HashSet::with_capacity(64);
        self.recurse(
            id,
            max_depth,
            0,
            &mut out,
            &mut prefix,
            &mut visited,
            Direction::Callees,
            local_only,
            show_ids,
        );
        out
    }

    /// One recursive descent; direction selects the edge map and target field.
    /// `prefix` is an accumulator passed down — each level pushes its own
    /// segment and truncates on the way back, so we allocate zero strings per
    /// visited node.
    #[allow(clippy::too_many_arguments)]
    fn recurse(
        &self,
        id: u32,
        max_depth: usize,
        depth: usize,
        out: &mut String,
        prefix: &mut String,
        visited: &mut HashSet<u32>,
        direction: Direction,
        local_only: bool,
        show_ids: bool,
    ) {
        if depth >= max_depth || !visited.insert(id) {
            return;
        }

        let edges_for_id: &[Edge] = match direction {
            Direction::Callers => self
                .index
                .reverse_edges
                .get(&id)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            Direction::Callees => self
                .index
                .forward_edges
                .get(&id)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        };

        let mut seen_external = HashSet::with_capacity(edges_for_id.len());
        let filtered: Vec<&Edge> = edges_for_id
            .iter()
            .filter(|e| e.kind == EdgeKind::Call)
            .filter(|e| {
                if local_only {
                    matches!(e.to, Target::Resolved(_))
                } else {
                    true
                }
            })
            .filter(|e| match &e.to {
                Target::External(name) => seen_external.insert(name.as_str()),
                Target::Resolved(_) | Target::Unresolved(_) => true,
            })
            .collect();

        let total = filtered.len();
        if total == 0 {
            return;
        }

        for (i, edge) in filtered.iter().enumerate() {
            let is_last = i + 1 == total;
            let branch = if is_last { "└── " } else { "├── " };
            let child_segment = if is_last { "    " } else { "│   " };

            out.push_str(prefix);
            out.push_str(branch);

            match direction {
                Direction::Callers => {
                    if let Target::Resolved(from_id) = edge.to {
                        if let Some(sym) = self.index.symbols.get(from_id as usize) {
                            if show_ids {
                                let _ = writeln!(
                                    out,
                                    "{}#{} {}",
                                    sym.kind.short(),
                                    from_id,
                                    sym.qualified
                                );
                            } else {
                                let _ = writeln!(out, "{} {}", sym.kind.short(), sym.qualified);
                            }
                            let saved = prefix.len();
                            prefix.push_str(child_segment);
                            self.recurse(
                                from_id,
                                max_depth,
                                depth + 1,
                                out,
                                prefix,
                                visited,
                                direction,
                                local_only,
                                show_ids,
                            );
                            prefix.truncate(saved);
                        } else {
                            out.push('\n');
                        }
                    } else {
                        out.push('\n');
                    }
                }
                Direction::Callees => match &edge.to {
                    Target::Resolved(to_id) => {
                        if let Some(sym) = self.index.symbols.get(*to_id as usize) {
                            if show_ids {
                                let _ = writeln!(
                                    out,
                                    "{}#{} {}",
                                    sym.kind.short(),
                                    to_id,
                                    sym.qualified
                                );
                            } else {
                                let _ = writeln!(out, "{} {}", sym.kind.short(), sym.qualified);
                            }
                            let saved = prefix.len();
                            prefix.push_str(child_segment);
                            self.recurse(
                                *to_id,
                                max_depth,
                                depth + 1,
                                out,
                                prefix,
                                visited,
                                direction,
                                local_only,
                                show_ids,
                            );
                            prefix.truncate(saved);
                        } else {
                            out.push('\n');
                        }
                    }
                    Target::External(name) => {
                        let _ = writeln!(out, "{}", name);
                    }
                    Target::Unresolved(name) => {
                        let _ = writeln!(out, "?{}", name);
                    }
                },
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Callers,
    Callees,
}
