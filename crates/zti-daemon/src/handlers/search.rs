use std::collections::HashMap;

use zti_protocol::request::{SearchMode, SearchReq};
use zti_protocol::response::{Response, SearchHit, SearchResults};
use zti_rerank::TurboReranker;
use zti_store::chunks_table::ChunkHit;

use crate::handlers::with_project;
use crate::state::DaemonState;

const APPENDIX_CAP: usize = 3;

pub async fn handle(req: &SearchReq, state: &DaemonState) -> Response {
    let result = with_project(state, &req.project_root, |project| async move {
        let pid =
            zti_common::ids::project_id(&std::path::Path::new(&req.project_root).canonicalize()?);

        let model_id = project
            .db
            .projects_table()
            .await?
            .get(&pid)
            .await?
            .and_then(|p| {
                if p.model_id.is_empty() {
                    None
                } else {
                    Some(p.model_id)
                }
            });

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
            let reranker = TurboReranker::new(engine.dim())?;
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
            )
            .await?
        };
        let hits = outcome.hits;

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
