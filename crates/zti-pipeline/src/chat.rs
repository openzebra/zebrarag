use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Result;

use zti_dsl::{EdgeKind, Kind, ProjectIndex, Target, LEGEND_LINE};
use zti_dsl::chunking::Chunk;
use zti_embed::EmbedEngine;
use zti_rerank::TurboReranker;

const KNN_OVERFETCH_MULT: usize = 3;
const DIVERSITY_PENALTY: f32 = 0.04;
const APPENDIX_CAP: usize = 8;
const APPENDIX_DEPTH: usize = 2;

pub fn run(
    engine: &EmbedEngine,
    chunks: &[Chunk],
    reranker: &TurboReranker,
    index: &ProjectIndex,
    top_k: usize,
) -> Result<()> {
    let renderer = ResponseRenderer::new(chunks);
    let mut rl = rustyline::DefaultEditor::new()?;
    println!();
    println!("zebra chat - type a question, :q to quit, Ctrl-D to exit");
    println!();

    while let Ok(line) = rl.readline("> ") {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == ":q" {
            break;
        }
        let _ = rl.add_history_entry(trimmed);

        let query_emb = match engine.embed_query(trimmed) {
            Ok(e) => e,
            Err(e) => {
                println!("  (embedding failed: {})", e);
                continue;
            }
        };

        let candidate_indices: Vec<usize> = (0..chunks.len()).collect();
        let mut ranked = reranker.rerank(&candidate_indices, &query_emb);
        ranked = diversify(ranked, chunks, index, top_k);

        if ranked.is_empty() {
            println!("  no results");
            continue;
        }

        println!();
        println!("{}", LEGEND_LINE);

        let match_ids: HashSet<u32> = ranked
            .iter()
            .map(|(idx, _)| chunks[*idx].sym_id)
            .collect();

        for (rank, (idx, score)) in ranked.iter().enumerate() {
            let chunk = match chunks.get(*idx) {
                Some(c) => c,
                None => continue,
            };
            println!("#{} {:.4} {}", rank + 1, score, chunk.qualified);
            print_block(&chunk.header, &chunk.body);
        }

        let appendix_ids = collect_appendix(
            &ranked,
            chunks,
            index,
            &match_ids,
            &renderer.chunk_by_sym,
            APPENDIX_DEPTH,
            APPENDIX_CAP,
        );

        if !appendix_ids.is_empty() {
            println!("--- APPENDIX ---");
            for &id in &appendix_ids {
                if let Some(&chunk_idx) = renderer.chunk_by_sym.get(&id) {
                    let chunk = &renderer.chunks[chunk_idx];
                    print_block(&chunk.header, &chunk.body);
                }
            }
        }
    }
    Ok(())
}

fn diversify(
    ranked: Vec<(usize, f32)>,
    chunks: &[Chunk],
    index: &ProjectIndex,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut parents_seen: HashMap<u32, usize> = HashMap::new();
    let mut diversified: Vec<(usize, f32)> = ranked
        .into_iter()
        .map(|(idx, score)| {
            let sym_id = chunks[idx].sym_id;
            let parent = index
                .symbols
                .get(sym_id as usize)
                .and_then(|s| s.parent);
            let parent = match parent {
                Some(p) => p,
                None => return (idx, score),
            };
            let count = parents_seen.entry(parent).or_insert(0);
            let adjusted = score - (*count as f32) * DIVERSITY_PENALTY;
            *count += 1;
            (idx, adjusted)
        })
        .collect();
    diversified.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    diversified.truncate(k);
    diversified
}

fn print_block(header: &str, body: &str) {
    for line in header.lines() {
        println!("  {}", line);
    }
    println!("  ---");
    for line in body.lines() {
        println!("  {}", line);
    }
}

fn collect_appendix(
    ranked: &[(usize, f32)],
    chunks: &[Chunk],
    index: &ProjectIndex,
    match_ids: &HashSet<u32>,
    chunk_by_sym: &HashMap<u32, usize>,
    max_depth: usize,
    cap: usize,
) -> Vec<u32> {
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(u32, usize)> = VecDeque::new();

    for &(idx, _) in ranked {
        let sym_id = chunks[idx].sym_id;
        if visited.insert(sym_id) {
            queue.push_back((sym_id, 0));
        }
    }

    let mut result = Vec::new();

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        for edge in index
            .edges
            .iter()
            .filter(|e| e.from == current && e.kind == EdgeKind::Call)
        {
            let Target::Resolved(rid) = edge.to else {
                continue;
            };
            if !visited.insert(rid) {
                continue;
            }
            if match_ids.contains(&rid) {
                continue;
            }
            if index.symbols.get(rid as usize).is_none() {
                continue;
            }
            if !chunk_by_sym.contains_key(&rid) {
                continue;
            }
            if result.len() < cap {
                result.push(rid);
            }
            queue.push_back((rid, depth + 1));
        }
    }

    result
}

struct ResponseRenderer<'a> {
    chunks: &'a [Chunk],
    chunk_by_sym: HashMap<u32, usize>,
}

impl<'a> ResponseRenderer<'a> {
    fn new(chunks: &'a [Chunk]) -> Self {
        let chunk_by_sym = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.sym_id, i))
            .collect();
        Self {
            chunks,
            chunk_by_sym,
        }
    }
}
