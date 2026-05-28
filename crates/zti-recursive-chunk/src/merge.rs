use std::collections::BinaryHeap;

use crate::atom::{AtomChunk, AtomCollector, DEFAULT_LANG_CONFIG, SynLangConfig, LineBreakLevel};
use crate::positions::{BytePos, compute_positions};
use crate::SubChunk;

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
) {
    let start = node.start_byte();
    let end = node.end_byte();

    if end - start <= min_atom {
        collector.curr_level = level;
        collector.add(start, end);
        return;
    }

    let mut cursor = node.walk();
    if !cursor.goto_first_child() {
        collector.curr_level = level + 1;
        collect_atoms(text, start, end, 0, min_atom, collector);
        return;
    }

    let mut prev_end = start;
    loop {
        let child = cursor.node();
        let cs = child.start_byte();

        if cs > prev_end {
            collector.curr_level = level + 1;
            collect_atoms(text, prev_end, cs, 0, min_atom, collector);
        }

        collect_atoms_ts(text, child, min_atom, level + 1, collector);

        prev_end = child.end_byte();

        if !cursor.goto_next_sibling() {
            break;
        }
    }

    if prev_end < end {
        collector.curr_level = level + 1;
        collect_atoms(text, prev_end, end, 0, min_atom, collector);
    }
}

/// Recursively split a byte range using progressively finer regex separators.
fn collect_atoms(
    text: &str,
    range_start: usize,
    range_end: usize,
    sep_id: usize,
    min_atom: usize,
    collector: &mut AtomCollector<'_>,
) {
    if range_end - range_start <= min_atom {
        collector.add(range_start, range_end);
        return;
    }
    let cfg: &SynLangConfig = &DEFAULT_LANG_CONFIG;
    if sep_id >= cfg.separator_regex.len() {
        collector.add(range_start, range_end);
        return;
    }
    let re = &cfg.separator_regex[sep_id];
    let fragment = &text[range_start..range_end];
    let mut cursor = range_start;
    for m in re.find_iter(fragment) {
        let sep_end = range_start + m.end();
        if cursor < sep_end {
            collect_atoms(text, cursor, range_start + m.start(), sep_id + 1, min_atom, collector);
        }
        cursor = sep_end;
    }
    if cursor < range_end {
        collect_atoms(text, cursor, range_end, sep_id + 1, min_atom, collector);
    }
}

fn lb_gap(boundary: &LineBreakLevel, internal: &LineBreakLevel) -> usize {
    if boundary.ord() < internal.ord() {
        internal.ord() - boundary.ord()
    } else {
        0
    }
}

fn merge_atoms(atoms: Vec<AtomChunk>, chunk_size: usize, chunk_overlap: usize, min_chunk: usize, text: &str) -> Vec<(BytePos, BytePos)> {
    struct Plan {
        start_idx: usize,
        prev_plan: usize,
        cost: usize,
        overlap_base: usize,
    }

    let n = atoms.len();
    let mut plans: Vec<Plan> = Vec::with_capacity(n);
    plans.push(Plan { start_idx: 0, prev_plan: 0, cost: 0, overlap_base: overlap_cost_base(text.len(), 0, chunk_overlap) });

    let mut heap: BinaryHeap<(std::cmp::Reverse<usize>, usize)> = BinaryHeap::new();
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
                heap.push((std::cmp::Reverse(plans[si].cost + plans[si].overlap_base), si));
                heap.peek().unwrap().1
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

    let mut out: Vec<(BytePos, BytePos)> = Vec::new();
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
    text_len.saturating_sub(offset).saturating_mul(MISSING_OVERLAP).checked_div(overlap).unwrap_or(0)
}

pub(crate) fn chunk_text_with_ts(
    source: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk: usize,
    root: tree_sitter::Node,
) -> Vec<SubChunk> {
    let min_atom = if chunk_overlap > 0 { chunk_overlap } else { min_chunk };

    let mut collector = AtomCollector {
        text: source,
        curr_level: 0,
        min_level: 0,
        chunks: Vec::with_capacity(source.len() / 32 + 1),
    };

    collect_atoms_ts(source, root, min_atom, 0, &mut collector);
    let atoms = collector.seal();
    let mut raw = merge_atoms(atoms, chunk_size, chunk_overlap, min_chunk, source);

    let all_pos: Vec<&mut BytePos> = raw.iter_mut().flat_map(|(s, e)| {
        std::iter::once(&mut *s).chain(std::iter::once(&mut *e))
    }).collect();
    compute_positions(source, all_pos);

    raw.into_iter().map(|(sp, ep)| {
        let s = sp.output.unwrap();
        let e = ep.output.unwrap();
        SubChunk {
            byte_start: sp.byte_offset,
            byte_end: ep.byte_offset,
            start_line: s.line,
            end_line: e.line,
        }
    }).collect()
}

pub(crate) fn chunk_text(
    source: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk: usize,
) -> Vec<SubChunk> {
    let min_atom = if chunk_overlap > 0 { chunk_overlap } else { min_chunk };

    let mut collector = AtomCollector {
        text: source,
        curr_level: 0,
        min_level: 0,
        chunks: Vec::with_capacity(source.len() / 32 + 1),
    };

    collect_atoms(source, 0, source.len(), 0, min_atom, &mut collector);
    let atoms = collector.seal();
    let mut raw = merge_atoms(atoms, chunk_size, chunk_overlap, min_chunk, source);

    let all_pos: Vec<&mut BytePos> = raw.iter_mut().flat_map(|(s, e)| {
        std::iter::once(&mut *s).chain(std::iter::once(&mut *e))
    }).collect();
    compute_positions(source, all_pos);

    raw.into_iter().map(|(sp, ep)| {
        let s = sp.output.unwrap();
        let e = ep.output.unwrap();
        SubChunk {
            byte_start: sp.byte_offset,
            byte_end: ep.byte_offset,
            start_line: s.line,
            end_line: e.line,
        }
    }).collect()
}
