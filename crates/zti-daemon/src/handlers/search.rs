use std::collections::HashMap;

use zti_protocol::request::{SearchMode, SearchReq};
use zti_protocol::response::{Response, SearchHit, SearchResults};
use zti_store::chunks_table::ChunkHit;

use crate::handlers::with_project;
use crate::state::DaemonState;

const APPENDIX_CAP: usize = 3;

pub async fn handle(req: &SearchReq, state: &DaemonState) -> Response {
    let result = with_project(state, &req.project_root, |project| async move {
        let pid =
            zti_common::ids::project_id(&std::path::Path::new(&req.project_root).canonicalize()?);

        let project_row = project
            .db
            .projects_table()
            .await?
            .get(&pid)
            .await?
            .ok_or_else(|| anyhow::anyhow!("project not indexed"))?;
        if project_row.index_version < zti_store::projects_table::INDEX_FORMAT_VERSION {
            anyhow::bail!("project index is stale; run: zebraindex index -r {}", req.project_root);
        }
        let model_id = if project_row.model_id.is_empty() {
            None
        } else {
            Some(project_row.model_id)
        };

        let engine = match model_id.as_deref() {
            Some(mid) => state.engine_for_model(mid).await?,
            None => state.primary_engine(),
        };

        let opts = zti_pipeline::search::SearchOpts {
            limit: req.limit,
            languages: req.languages.as_deref(),
            path_glob: req.path_glob.as_deref(),
            include_tests: req.include_tests,
        };

        let query_emb = match req.mode {
            SearchMode::Query => engine.embed_query_async(&req.query).await?,
            SearchMode::Passage => engine.embed_passage_async(&req.query).await?,
        };

        let cached_params = if req.exhaustive {
            None
        } else if let Some(params) = project.search_params.read().await.as_ref() {
            Some(std::sync::Arc::clone(params))
        } else {
            let parsed = project_row
                .search_params
                .as_deref()
                .and_then(|params| toml::from_str(params).ok())
                .unwrap_or_else(|| {
                    zti_ann::choose_method(
                        project_row.total_chunks as usize,
                        engine.dim(),
                        state.hardware.as_ref(),
                        None,
                    )
                });
            let parsed = std::sync::Arc::new(parsed);
            *project.search_params.write().await = Some(std::sync::Arc::clone(&parsed));
            Some(parsed)
        };

        let outcome = if req.exhaustive {
            zti_pipeline::search::search_exhaustive(
                &req.query,
                &query_emb,
                &engine,
                &project.db,
                &pid,
                &opts,
            )
            .await?
        } else {
            let reranker = state.reranker.get(engine.dim()).await?;
            zti_pipeline::search::search(
                &req.query,
                &query_emb,
                &engine,
                &project.db,
                &reranker,
                &state.ann,
                &state.turbo,
                &pid,
                &opts,
                cached_params.as_deref(),
                Some(project_row.total_chunks as usize),
            )
            .await?
        };
        let mut hits = outcome.hits;
        dedup_overlapping_hits(&mut hits);

        let chunks_table = project.db.chunks_table(engine.dim()).await?;

        // Walk `hits` once to collect (a) sym_ids already in the top-N (so the
        // appendix dedupe HashSet is seeded) and (b) the appendix candidate
        // ids — both reads need only borrows. After this scan we are free to
        // consume `hits` by value and move every `ChunkHit` into `search_hits`
        // without cloning the heap-allocated String fields.
        let mut seen: std::collections::HashSet<u32> =
            std::collections::HashSet::with_capacity(hits.len() + APPENDIX_CAP);
        for h in &hits {
            seen.insert(h.chunk.sym_id);
        }
        let mut appendix_ids: Vec<u32> = Vec::with_capacity(APPENDIX_CAP);
        'outer: for h in &hits {
            for &sid in &h.chunk.appendix_sym_ids {
                if appendix_ids.len() >= APPENDIX_CAP {
                    break 'outer;
                }
                if seen.insert(sid) {
                    appendix_ids.push(sid);
                }
            }
        }

        let search_hits: Vec<SearchHit> = hits
            .into_iter()
            .map(|h| chunk_to_hit(h.chunk, h.score, &req.project_root))
            .collect();

        let appendix = if appendix_ids.is_empty() {
            Vec::with_capacity(0)
        } else {
            let rows = chunks_table.get_by_sym_ids(&appendix_ids).await?;
            // Move rows into a sym_id → ChunkHit map so the loop below can
            // pop owned ChunkHits via `.remove(sid)` — no .clone() on the
            // heap String fields.
            let mut by_sym: HashMap<u32, ChunkHit> = HashMap::with_capacity(rows.len());
            for r in rows {
                by_sym.entry(r.sym_id).or_insert(r);
            }
            let mut out: Vec<SearchHit> = Vec::with_capacity(appendix_ids.len());
            for sid in &appendix_ids {
                if let Some(c) = by_sym.remove(sid) {
                    out.push(chunk_to_hit(c, 0.0, &req.project_root));
                }
            }
            out
        };

        let total = search_hits.len();

        Ok(SearchResults {
            hits: search_hits,
            appendix,
            total,
        })
    })
    .await;

    Response::Search(result)
}

fn dedup_overlapping_hits(hits: &mut Vec<zti_pipeline::search::Hit>) {
    let mut kept = 0usize;
    for i in 0..hits.len() {
        let current = &hits[i];
        let is_duplicate = hits[..kept].iter().any(|kept_hit| {
            kept_hit.chunk.file_path == current.chunk.file_path
                && kept_hit.chunk.symbol_qualified == current.chunk.symbol_qualified
                && kept_hit.chunk.start_line.max(current.chunk.start_line)
                    <= kept_hit.chunk.end_line.min(current.chunk.end_line)
        });
        if !is_duplicate {
            hits.swap(kept, i);
            kept = kept.saturating_add(1);
        }
    }
    hits.truncate(kept);
}

fn chunk_to_hit(mut c: ChunkHit, score: f32, project_root: &str) -> SearchHit {
    if c.file_path.starts_with(project_root) {
        c.file_path.drain(..project_root.len());
    }
    let lead_slashes = c.file_path.bytes().take_while(|b| *b == b'/').count();
    if lead_slashes > 0 {
        c.file_path.drain(..lead_slashes);
    }
    SearchHit {
        chunk_id: c.chunk_id,
        file_path: c.file_path,
        symbol_qualified: c.symbol_qualified,
        symbol_kind: c.symbol_kind,
        sym_id: c.sym_id,
        start_line: c.start_line,
        end_line: c.end_line,
        content: c.content,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::dedup_overlapping_hits;
    use zti_common::{chunk_strategy::ChunkStrategy, file_type::FileType};
    use zti_pipeline::search::Hit;
    use zti_store::chunks_table::ChunkHit;

    fn hit(file_path: &str, symbol_qualified: &str, start_line: u32, end_line: u32) -> Hit {
        Hit {
            chunk: ChunkHit {
                chunk_id: [0u8; 16],
                file_path: file_path.to_string(),
                file_type: FileType::Source,
                symbol_qualified: symbol_qualified.to_string(),
                symbol_kind: String::from("function"),
                sym_id: 0,
                sub_chunk_idx: 0,
                total_sub_chunks: 1,
                chunk_strategy: ChunkStrategy::Symbol,
                parent_sym_id: None,
                appendix_sym_ids: Vec::with_capacity(0),
                start_line,
                end_line,
                content: String::new(),
                turbo_code: Vec::with_capacity(0),
                score: 0.0,
            },
            score: 1.0,
        }
    }

    #[test]
    fn dedup_overlapping_hits_keeps_distinct_symbols_and_ranges() {
        let mut hits = vec![
            hit("src/lib.rs", "top_k", 56, 68),
            hit("src/lib.rs", "top_k", 61, 73),
            hit("src/lib.rs", "top_k", 80, 90),
            hit("src/lib.rs", "other", 61, 73),
        ];
        dedup_overlapping_hits(&mut hits);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].chunk.symbol_qualified, "top_k");
        assert_eq!(hits[0].chunk.start_line, 56);
        assert_eq!(hits[1].chunk.start_line, 80);
        assert_eq!(hits[2].chunk.symbol_qualified, "other");
    }
}
