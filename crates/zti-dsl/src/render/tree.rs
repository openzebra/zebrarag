use crate::model::{Edge, EdgeKind, ProjectIndex, Target};

pub struct AsciiTreeRenderer<'a> {
    index: &'a ProjectIndex,
}

impl<'a> AsciiTreeRenderer<'a> {
    pub fn new(index: &'a ProjectIndex) -> Self {
        Self { index }
    }

    pub fn render_callers(&self, id: u32, max_depth: usize) -> String {
        let mut out = String::new();
        let sym = match self.index.symbols.get(id as usize) {
            Some(s) => s,
            None => return format!("Symbol {} not found\n", id),
        };
        out.push_str(&format!("{}#{} {} (callers)\n", sym.kind.short(), id, sym.qualified));
        self.render_callers_recursive(id, max_depth, 0, &mut out, &mut std::collections::HashSet::new());
        out
    }

    pub fn render_callees(&self, id: u32, max_depth: usize) -> String {
        let mut out = String::new();
        let sym = match self.index.symbols.get(id as usize) {
            Some(s) => s,
            None => return format!("Symbol {} not found\n", id),
        };
        out.push_str(&format!("{}#{} {} (callees)\n", sym.kind.short(), id, sym.qualified));
        self.render_callees_recursive(id, max_depth, 0, &mut out, &mut std::collections::HashSet::new());
        out
    }

    fn render_callers_recursive(
        &self,
        id: u32,
        max_depth: usize,
        depth: usize,
        out: &mut String,
        visited: &mut std::collections::HashSet<u32>,
    ) {
        if depth >= max_depth || !visited.insert(id) {
            return;
        }

        if let Some(edges) = self.index.reverse_edges.get(&id) {
            let call_edges: Vec<&Edge> = edges.iter().filter(|e| e.kind == EdgeKind::Call).collect();
            for (i, edge) in call_edges.iter().enumerate() {
                let is_last = i == call_edges.len() - 1;
                let prefix = if is_last { "└── " } else { "├── " };
                let child_prefix = if is_last { "    " } else { "│   " };

                let indent = (0..depth).map(|_| child_prefix).collect::<String>();
                out.push_str(&indent);
                out.push_str(prefix);

                if let Target::Resolved(from_id) = edge.to
                    && let Some(sym) = self.index.symbols.get(from_id as usize) {
                        out.push_str(&format!("{}#{} {}\n", sym.kind.short(), from_id, sym.qualified));
                        self.render_callers_recursive(from_id, max_depth, depth + 1, out, visited);
                    }
            }
        }
    }

    fn render_callees_recursive(
        &self,
        id: u32,
        max_depth: usize,
        depth: usize,
        out: &mut String,
        visited: &mut std::collections::HashSet<u32>,
    ) {
        if depth >= max_depth || !visited.insert(id) {
            return;
        }

        let callees: Vec<&Edge> = self.index.edges
            .iter()
            .filter(|e| e.from == id && e.kind == EdgeKind::Call)
            .collect();

        for (i, edge) in callees.iter().enumerate() {
            let is_last = i == callees.len() - 1;
            let prefix = if is_last { "└── " } else { "├── " };
            let child_prefix = if is_last { "    " } else { "│   " };

            let indent = (0..depth).map(|_| child_prefix).collect::<String>();
            out.push_str(&indent);
            out.push_str(prefix);

            match &edge.to {
                Target::Resolved(to_id) => {
                    if let Some(sym) = self.index.symbols.get(*to_id as usize) {
                        out.push_str(&format!("{}#{} {}\n", sym.kind.short(), to_id, sym.qualified));
                        self.render_callees_recursive(*to_id, max_depth, depth + 1, out, visited);
                    }
                }
                Target::External(name) => {
                    out.push_str(&format!("*{}\n", name));
                }
                Target::Unresolved(name) => {
                    out.push_str(&format!("?{}\n", name));
                }
            }
        }
    }
}
