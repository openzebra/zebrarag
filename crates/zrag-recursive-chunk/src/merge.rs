use std::collections::BinaryHeap;

use crate::SubChunk;
use crate::atom::{AtomChunk, AtomCollector, DEFAULT_LANG_CONFIG, LineBreakLevel, SynLangConfig};
use crate::positions::{BytePos, OutputPos, compute_positions};

const SYNTAX_GAP: usize = 512;
const MISSING_OVERLAP: usize = 512;
const PER_LB: usize = 64;
const TOO_SMALL: usize = 1048576;

/// Walk a tree-sitter AST node, collecting atoms from its children.
/// Gaps between siblings and terminal nodes fall through to regex splitting.
fn collect_atoms_ts(
    text: &str,
    node: tree_sitter::Node,
    min_atom: usize,
    level: usize,
    collector: &mut AtomCollector,
    terminal_kinds: &[u16],
) {
    let start = node.start_byte();
    let end = node.end_byte();

    if end - start <= min_atom {
        collector.curr_level = level;
        collector.add(start, end);
        return;
    }

    if terminal_kinds.contains(&node.kind_id()) {
        collector.curr_level = level;
        collector.add(start, end);
        return;
    }

    let mut cursor = node.walk();
    if !cursor.goto_first_child() {
        collect_atoms(text, start, end, level, 0, min_atom, collector);
        return;
    }

    let mut prev_end = start;
    loop {
        let child = cursor.node();
        let cs = child.start_byte();

        if cs > prev_end {
            collect_atoms(text, prev_end, cs, level, 0, min_atom, collector);
        }

        collect_atoms_ts(text, child, min_atom, level + 1, collector, terminal_kinds);

        prev_end = child.end_byte();

        if !cursor.goto_next_sibling() {
            break;
        }
    }

    if prev_end < end {
        collect_atoms(text, prev_end, end, level, 0, min_atom, collector);
    }
}

/// Recursively split a byte range using progressively finer regex separators.
/// `base_level` offsets syntax levels so gaps inside tree-sitter nodes at depth D
/// produce atoms on the same scale as sibling TS nodes (both at D+1).
fn collect_atoms(
    text: &str,
    range_start: usize,
    range_end: usize,
    base_level: usize,
    sep_id: usize,
    min_atom: usize,
    collector: &mut AtomCollector<'_>,
) {
    if range_end - range_start <= min_atom {
        collector.curr_level = base_level + sep_id + 1;
        collector.add(range_start, range_end);
        return;
    }
    let cfg: &SynLangConfig = &DEFAULT_LANG_CONFIG;
    if sep_id >= cfg.separator_regex.len() {
        collector.curr_level = base_level + sep_id + 1;
        collector.add(range_start, range_end);
        return;
    }
    let re = &cfg.separator_regex[sep_id];
    let fragment = &text[range_start..range_end];
    let mut cursor = range_start;
    for m in re.find_iter(fragment) {
        let sep_end = range_start + m.end();
        if cursor < sep_end {
            collect_atoms(
                text,
                cursor,
                range_start + m.start(),
                base_level,
                sep_id + 1,
                min_atom,
                collector,
            );
        }
        cursor = sep_end;
    }
    if cursor < range_end {
        collect_atoms(
            text,
            cursor,
            range_end,
            base_level,
            sep_id + 1,
            min_atom,
            collector,
        );
    }
    let lvl = base_level + sep_id + 1;
    if lvl < collector.min_level {
        collector.min_level = lvl;
    }
    collector.curr_level = lvl;
}

fn lb_gap(boundary: &LineBreakLevel, internal: &LineBreakLevel) -> usize {
    if boundary.ord() < internal.ord() {
        internal.ord() - boundary.ord()
    } else {
        0
    }
}

fn merge_atoms(
    atoms: Vec<AtomChunk>,
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk: usize,
    text: &str,
) -> Vec<(BytePos, BytePos)> {
    struct Plan {
        start_idx: usize,
        prev_plan: usize,
        cost: usize,
        overlap_base: usize,
    }

    let n = atoms.len();
    let mut plans: Vec<Plan> = Vec::with_capacity(n);
    plans.push(Plan {
        start_idx: 0,
        prev_plan: 0,
        cost: 0,
        overlap_base: overlap_cost_base(text.len(), 0, chunk_overlap),
    });

    let mut heap: BinaryHeap<(std::cmp::Reverse<usize>, usize)> = BinaryHeap::with_capacity(n);
    let mut gap_cache = vec![0usize];

    for i in 0..n.saturating_sub(1) {
        let mut best = usize::MAX;
        let mut arg_start = 0usize;
        let mut arg_prev = 0usize;
        let mut si = i;

        let end_syn = atoms[i + 1].boundary_syntax_level;
        let end_lb = &atoms[i + 1].boundary_lb_level;

        let mut int_syn = usize::MAX;
        let mut int_lb = LineBreakLevel::Inline;

        loop {
            let sc = &atoms[si];
            let size = atoms[i].byte_end - sc.byte_start;

            let mut cost = 0usize;

            // syntax gap cost
            if sc.boundary_syntax_level > int_syn {
                let gap = sc.boundary_syntax_level - int_syn;
                for j in gap_cache.len()..=gap {
                    gap_cache.push(gap_cache[j - 1] + SYNTAX_GAP / j);
                }
                cost += gap_cache[gap];
            }
            if end_syn > int_syn {
                let gap = end_syn - int_syn;
                for j in gap_cache.len()..=gap {
                    gap_cache.push(gap_cache[j - 1] + SYNTAX_GAP / j);
                }
                cost += gap_cache[gap];
            }

            // line break gap
            cost += (lb_gap(&sc.boundary_lb_level, &int_lb) + lb_gap(end_lb, &int_lb)) * PER_LB;

            if size < min_chunk {
                cost += TOO_SMALL;
            }

            if size > chunk_size {
                if best == usize::MAX {
                    best = cost + plans[si].cost;
                    arg_start = si;
                    arg_prev = si;
                }
                break;
            }

            let prev = if chunk_overlap > 0 {
                while let Some(&(_, idx)) = heap.peek() {
                    if atoms[idx].byte_end - sc.byte_start <= chunk_overlap {
                        break;
                    }
                    heap.pop();
                }
                heap.push((
                    std::cmp::Reverse(plans[si].cost + plans[si].overlap_base),
                    si,
                ));
                match heap.peek() {
                    Some(&(_, idx)) => idx,
                    None => si,
                }
            } else {
                si
            };

            let p = &plans[prev];
            cost += p.cost;

            if chunk_overlap == 0 {
                cost += MISSING_OVERLAP / 2;
            } else {
                let sb = overlap_cost_base(text.len(), sc.byte_start, chunk_overlap);
                cost += if p.overlap_base < sb {
                    MISSING_OVERLAP + p.overlap_base - sb
                } else {
                    MISSING_OVERLAP
                };
            }

            if cost < best {
                best = cost;
                arg_start = si;
                arg_prev = prev;
            }

            if si == 0 {
                break;
            }
            si -= 1;
            int_syn = int_syn.min(sc.boundary_syntax_level);
            int_lb = int_lb.max(sc.internal_lb_level);
        }

        plans.push(Plan {
            start_idx: arg_start,
            prev_plan: arg_prev,
            cost: best,
            overlap_base: overlap_cost_base(text.len(), atoms[i].byte_end, chunk_overlap),
        });
        heap.clear();
    }

    let mut out: Vec<(BytePos, BytePos)> = Vec::with_capacity(plans.len());
    let mut pi = plans.len() - 1;
    while pi > 0 {
        let p = &plans[pi];
        out.push((
            BytePos::new(atoms[p.start_idx].byte_start),
            BytePos::new(atoms[pi - 1].byte_end),
        ));
        pi = p.prev_plan;
    }
    out.reverse();
    out
}

fn overlap_cost_base(text_len: usize, offset: usize, overlap: usize) -> usize {
    text_len
        .saturating_sub(offset)
        .saturating_mul(MISSING_OVERLAP)
        .checked_div(overlap)
        .unwrap_or(0)
}

fn finish_chunks(
    source: &str,
    collector: AtomCollector,
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk: usize,
) -> Vec<SubChunk> {
    let atoms = collector.seal();
    let mut raw = merge_atoms(atoms, chunk_size, chunk_overlap, min_chunk, source);

    let all_pos: Vec<&mut BytePos> = raw
        .iter_mut()
        .flat_map(|(s, e)| [s as &mut BytePos, e as &mut BytePos])
        .collect();
    compute_positions(source, all_pos);

    raw.into_iter()
        .map(|(sp, ep)| {
            let s = match sp.output {
                Some(o) => o,
                None => OutputPos { line: 1 },
            };
            let e = match ep.output {
                Some(o) => o,
                None => OutputPos { line: 1 },
            };
            SubChunk {
                byte_start: sp.byte_offset,
                byte_end: ep.byte_offset,
                start_line: s.line,
                end_line: e.line,
            }
        })
        .collect()
}

fn min_atom_size(chunk_overlap: usize, min_chunk: usize) -> usize {
    if chunk_overlap > 0 {
        chunk_overlap
    } else {
        min_chunk
    }
}

pub(crate) fn chunk_text_with_ts(
    source: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk: usize,
    root: tree_sitter::Node,
    terminal_kinds: &[u16],
) -> Vec<SubChunk> {
    let min_atom = min_atom_size(chunk_overlap, min_chunk);
    let mut collector = AtomCollector::new(source);
    collect_atoms_ts(source, root, min_atom, 0, &mut collector, terminal_kinds);
    finish_chunks(source, collector, chunk_size, chunk_overlap, min_chunk)
}

pub(crate) fn chunk_text(
    source: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk: usize,
) -> Vec<SubChunk> {
    let min_atom = min_atom_size(chunk_overlap, min_chunk);
    let mut collector = AtomCollector::new(source);
    collect_atoms(source, 0, source.len(), 0, 0, min_atom, &mut collector);
    finish_chunks(source, collector, chunk_size, chunk_overlap, min_chunk)
}

#[cfg(test)]
mod tests_merge {
    use super::*;

    #[test]
    fn test_split_basic() {
        let source = "Linea 1.\nLinea 2.\n\nLinea 3.";
        let chunks = chunk_text(source, 15, 0, 5);
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            &source[chunks[0].byte_start..chunks[0].byte_end],
            "Linea 1."
        );
        assert_eq!(
            &source[chunks[1].byte_start..chunks[1].byte_end],
            "Linea 2."
        );
        assert_eq!(
            &source[chunks[2].byte_start..chunks[2].byte_end],
            "Linea 3."
        );
    }

    #[test]
    fn test_split_long_text() {
        let source = "A very very long text that needs to be split.";
        let chunks = chunk_text(source, 20, 0, 12);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            let t = &source[chunk.byte_start..chunk.byte_end];
            assert!(t.len() <= 20, "Chunk too long: '{}'", t);
        }
    }

    #[test]
    fn test_split_with_overlap() {
        let source = "This is a test text that is a bit longer to see how the overlap works.";
        let chunks = chunk_text(source, 20, 5, 10);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            let t = &source[chunk.byte_start..chunk.byte_end];
            assert!(t.len() <= 25, "Chunk too long: '{}'", t);
        }
    }

    #[test]
    fn test_split_trims_whitespace() {
        let source = "  \n First chunk  \n\n  Second chunk with spaces at the end    \n";
        let chunks = chunk_text(source, 30, 0, 10);
        assert!(!chunks.is_empty());
        let first = &source[chunks[0].byte_start..chunks[0].byte_end];
        assert!(
            !first.starts_with("  "),
            "First chunk should not start with spaces, got: '{}'",
            first
        );
    }

    /// Returns `(byte_start, byte_end, boundary_syntax_level)` for every atom
    /// produced by `collect_atoms_ts` + `seal` — used by differential tests.
    fn atoms_for_ts(source: &str, min_atom: usize) -> Vec<(usize, usize, usize)> {
        let tree = parse_rust(source);
        let mut collector = AtomCollector::new(source);
        collect_atoms_ts(source, tree.root_node(), min_atom, 0, &mut collector, &[]);
        let atoms = collector.seal();
        atoms
            .into_iter()
            .map(|a| (a.byte_start, a.byte_end, a.boundary_syntax_level))
            .collect()
    }

    fn parse_rust(source: &str) -> tree_sitter::Tree {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        parser.parse(source, None).unwrap()
    }

    #[test]
    fn test_ts_depth_offset_deep_gap() {
        // Blank-line gap inside `fn` at depth 2.
        // Cocoindex ground truth (tree-sitter-rust 0.24.0):
        //   atom[5]  "    let" (first let)         byte 12-19 bsl=4
        //   atom[10] "    let" (second let, after gap) byte 28-35 bsl=4
        //   sentinel bsl=0
        let source = "fn main() {\n    let a = 1;\n\n    let b = 2;\n}\n";
        let atoms = atoms_for_ts(source, 1);

        // Find the two let declarations by byte position.
        // First: source contains "    let" at bytes 12-19
        // Second: source contains "    let" at bytes 28-35
        let first_let_idx = atoms.iter().position(|&(s, e, _)| s == 12 && e == 19);
        let second_let_idx = atoms.iter().position(|&(s, e, _)| s == 28 && e == 35);

        let first_bsl = first_let_idx.map(|i| atoms[i].2);
        let second_bsl = second_let_idx.map(|i| atoms[i].2);

        assert!(
            first_bsl.is_some(),
            "first let_declaration atom (byte 12-19) not found in atoms"
        );
        assert!(
            second_bsl.is_some(),
            "second let_declaration atom (byte 28-35) not found in atoms"
        );

        let first_bsl = first_bsl.unwrap();
        let second_bsl = second_bsl.unwrap();

        // Both let declarations are at the same TS depth (children of block).
        // Cocoindex gives bsl=4 for both. Without base_level offset,
        // min_level resets to ~1 after the gap → second_bsl would be ~1.
        assert_eq!(
            first_bsl, second_bsl,
            "both let declarations should have same bsl (same TS depth); \
             first={}, second={}  (TS depth offset bug: gap resets min_level)",
            first_bsl, second_bsl,
        );
        // Cocoindex gives bsl=4; zebra is uniform −1 (no Once root frame).
        // The invariant is equality, not the absolute value.
        assert_eq!(first_bsl, 3, "expected bsl=3 (zebra = cocoindex − 1)");

        // Sentinel must be 0 (cocoindex ground truth, enforced by finish_chunks).
        let sentinel = atoms.last().unwrap();
        assert_eq!(sentinel.2, 0, "sentinel boundary_syntax_level must be 0");
    }

    #[test]
    fn test_split_with_rust_language() {
        let source = r#"
fn main() {
    println!("Hello");
}

fn other() {
    let x = 1;
}
"#;
        let tree = parse_rust(source);
        let root = tree.root_node();
        let chunks = chunk_text_with_ts(source, 50, 0, 20, root, &[]);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_split_positions() {
        let source = "Chunk1\n\nChunk2";
        let chunks = chunk_text(source, 10, 0, 5);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[1].start_line, 3);
    }

    #[test]
    fn terminal_kinds_prevent_regex_inside_comment() {
        let source = "/*\npara 1\n\npara 2\n\npara 3\n*/\nfn f() {}";
        let tree = parse_rust(source);
        let ts_lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        let block_comment_id = ts_lang.id_for_node_kind("block_comment", true);

        let mut collector_no_term = AtomCollector::new(source);
        collect_atoms_ts(source, tree.root_node(), 5, 0, &mut collector_no_term, &[]);
        let atoms_no_term = collector_no_term.seal();

        let mut collector_term = AtomCollector::new(source);
        collect_atoms_ts(
            source,
            tree.root_node(),
            5,
            0,
            &mut collector_term,
            &[block_comment_id],
        );
        let atoms_term = collector_term.seal();

        assert!(
            atoms_term.len() < atoms_no_term.len(),
            "terminal kinds should produce fewer atoms ({} vs {})",
            atoms_term.len(),
            atoms_no_term.len(),
        );
    }

    #[test]
    fn terminal_kinds_empty_no_effect() {
        let source = "fn f() { let x = 1; }";
        let tree = parse_rust(source);

        let mut collector1 = AtomCollector::new(source);
        collect_atoms_ts(source, tree.root_node(), 100, 0, &mut collector1, &[]);

        let mut collector2 = AtomCollector::new(source);
        collect_atoms_ts(source, tree.root_node(), 100, 0, &mut collector2, &[9999]);

        let atoms1 = collector1.seal();
        let atoms2 = collector2.seal();
        assert_eq!(
            atoms1.len(),
            atoms2.len(),
            "bogus terminal kind ID should not affect output"
        );
    }
}
