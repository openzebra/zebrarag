use std::borrow::Cow;
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ErrorData, ServiceExt, tool};
use tokio::sync::{Mutex, MutexGuard};
use zrag_ipc_client::Client;
use zrag_protocol::format_search_results;
use zrag_protocol::request::{DoctorReq, Request, SearchDepReq, SearchMode, SearchReq};
use zrag_protocol::response::{CheckStatus, Response, SearchResults};

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FileTreeParams {
    #[schemars(
        description = "Project name, index number, or root path. Use `projectList` to see available projects."
    )]
    pub project: String,
    #[schemars(
        description = "Optional glob pattern to filter files, e.g. \"**/*.rs\" or \"src/**/*.ts\"."
    )]
    pub path_glob: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchQueryParams {
    #[schemars(
        description = "What you're looking for, in natural language. Use descriptive phrases: \"polynomial inversion\" not \"invert\"."
    )]
    pub text: String,
    #[schemars(
        description = "Project name, index number, or root path. Auto-resolved when omitted."
    )]
    pub project: Option<String>,
    #[schemars(description = "Maximum results to return (default: 5).")]
    pub limit: Option<usize>,
    #[schemars(description = "Glob pattern to filter files, e.g. \"**/*.rs\" or \"src/**/*.ts\".")]
    pub path_glob: Option<String>,
    #[schemars(description = "Language filter, e.g. [\"rust\", \"dart\"].")]
    pub languages: Option<Vec<String>>,
    #[schemars(description = "Include test files in results (default: false).")]
    pub include_tests: Option<bool>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchPassageParams {
    #[schemars(
        description = "A code snippet, error message, or descriptive paragraph to find similar implementations."
    )]
    pub text: String,
    #[schemars(
        description = "Project name, index number, or root path. Auto-resolved when omitted."
    )]
    pub project: Option<String>,
    #[schemars(description = "Maximum results to return (default: 5).")]
    pub limit: Option<usize>,
    #[schemars(description = "Glob pattern to filter files, e.g. \"**/*.rs\" or \"src/**/*.ts\".")]
    pub path_glob: Option<String>,
    #[schemars(description = "Language filter, e.g. [\"rust\", \"dart\"].")]
    pub languages: Option<Vec<String>>,
    #[schemars(description = "Include test files in results (default: false).")]
    pub include_tests: Option<bool>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchDepParams {
    #[schemars(
        description = "Symbol, type, or function name. Bare (\"connect\"), scoped \
        (\"network::connect\"), or fully-qualified (\"myapp::network::connect\"). Use `::` \
        separators in any language."
    )]
    pub name: String,
    #[schemars(
        description = "Project name, index, or root path. Auto-resolved when omitted. To learn \
        an external dependency, index its source as a project first, then target it here."
    )]
    pub project: Option<String>,
    #[schemars(description = "Call-graph depth for callers/callees (default 2).")]
    pub depth: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DoctorParams {
    #[schemars(
        description = "Project name, index number, or root path. If omitted, runs general diagnostics."
    )]
    pub project: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectListParams {}

struct SearchCall<'a> {
    text: String,
    project: Option<&'a str>,
    limit: Option<usize>,
    path_glob: Option<String>,
    languages: Option<Vec<String>>,
    include_tests: Option<bool>,
    mode: SearchMode,
}

const POOL_SIZE: usize = 4;

struct DaemonPool {
    slots: [Mutex<Option<Client>>; POOL_SIZE],
    next: AtomicUsize,
}

struct PoolGuard<'a> {
    guard: MutexGuard<'a, Option<Client>>,
}

impl Default for DaemonPool {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Mutex::new(None)),
            next: AtomicUsize::new(0),
        }
    }
}

impl DaemonPool {
    async fn lease(&self) -> Result<PoolGuard<'_>, ErrorData> {
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for offset in 0..POOL_SIZE {
            let slot = (start + offset) % POOL_SIZE;
            let Some(mutex) = self.slots.get(slot) else {
                continue;
            };
            if let Ok(guard) = mutex.try_lock() {
                return Self::connect_if_needed(guard).await;
            }
        }
        let slot = start % POOL_SIZE;
        let mutex = self
            .slots
            .get(slot)
            .ok_or_else(|| internal_err("daemon pool slot missing".into()))?;
        let guard = mutex.lock().await;
        Self::connect_if_needed(guard).await
    }

    async fn connect_if_needed(
        mut guard: MutexGuard<'_, Option<Client>>,
    ) -> Result<PoolGuard<'_>, ErrorData> {
        if guard.is_none() {
            let mut client =
                Client::connect(Duration::from_secs(10), None, None, None, None, None, None)
                    .await
                    .map_err(|e| internal_err(format!("daemon connect: {e}")))?;
            client
                .handshake()
                .await
                .map_err(|e| internal_err(format!("handshake: {e}")))?;
            *guard = Some(client);
        }
        Ok(PoolGuard { guard })
    }
}

impl PoolGuard<'_> {
    fn client_mut(&mut self) -> Result<&mut Client, ErrorData> {
        self.guard
            .as_mut()
            .ok_or_else(|| internal_err("daemon not initialized".into()))
    }

    fn poison(&mut self) {
        *self.guard = None;
    }
}

#[derive(Clone)]
struct ZebraMcpServer {
    pool: Arc<DaemonPool>,
    indexed_projects_roots: String,
}

impl ZebraMcpServer {
    fn new(indexed_projects_roots: String) -> Self {
        Self {
            pool: Arc::new(DaemonPool::default()),
            indexed_projects_roots,
        }
    }

    async fn send<T>(
        &self,
        req: &Request,
        extract: impl FnOnce(Response) -> Result<T, ErrorData>,
    ) -> Result<T, ErrorData> {
        let mut lease = self.pool.lease().await?;
        match lease.client_mut()?.request(req).await {
            Ok(resp) => extract(resp),
            Err(e) => {
                lease.poison();
                Err(internal_err(format!("IPC lost, retry: {e}")))
            }
        }
    }

    async fn send_search(&self, req: &Request) -> Result<SearchResults, ErrorData> {
        self.send(req, |resp| match resp {
            Response::Search(Ok(results)) => Ok(results),
            Response::Search(Err(e)) => Err(internal_err(e.message)),
            other => Err(internal_err(format!("unexpected response: {other:?}"))),
        })
        .await
    }

    async fn do_search(&self, call: SearchCall<'_>) -> Result<CallToolResult, ErrorData> {
        let project_root = zrag_store::resolve_project(call.project)
            .await
            .map_err(|e| internal_err(format!("{e}")))?;
        let limit = call.limit.unwrap_or(5);

        let req = SearchReq {
            project_root,
            query: call.text,
            limit,
            offset: None,
            languages: call.languages,
            path_glob: call.path_glob,
            refresh_index: false,
            exhaustive: false,
            include_tests: call.include_tests.unwrap_or(false),
            mode: call.mode,
        };

        let mut wire_req = Request::Search(req);
        let results = self.send_search(&wire_req).await?;

        if !results.hits.is_empty() {
            return Ok(ok_text(format_search_results(&results)));
        }

        if let Request::Search(search_req) = &mut wire_req {
            search_req.exhaustive = true;
        }
        let retry_results = self.send_search(&wire_req).await?;

        Ok(ok_text(format_search_results(&retry_results)))
    }

    async fn send_search_dep(&self, req: SearchDepReq) -> Result<String, ErrorData> {
        self.send(&Request::DslSearchDep(req), |resp| match resp {
            Response::DslSearchDep(Ok(body)) => Ok(body.text),
            Response::DslSearchDep(Err(e)) => Err(internal_err(e.message)),
            other => Err(internal_err(format!("unexpected response: {other:?}"))),
        })
        .await
    }
}

fn match_file(file_path: &str, root: &str, matcher: Option<&globset::GlobMatcher>) -> bool {
    let Some(m) = matcher else { return true };
    let rel = file_path
        .strip_prefix(root)
        .unwrap_or(file_path)
        .trim_start_matches('/');
    m.is_match(rel) || m.is_match(file_path)
}

fn ok_text(text: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text.into())])
}

fn internal_err(msg: String) -> ErrorData {
    ErrorData::internal_error(msg, None)
}

fn format_project_table(projects: &[zrag_store::ProjectRow]) -> String {
    let mut out = String::with_capacity(projects.len() * 80);
    out.push_str("| # | Project | Root |\n");
    out.push_str("|---|---------|------|\n");
    for (i, p) in projects.iter().enumerate() {
        let name = std::path::Path::new(&p.root_path)
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_else(|| Cow::Borrowed(&p.root_path));
        let _ = writeln!(out, "| {} | {} | {} |", i + 1, name, p.root_path);
    }
    out
}

#[rmcp::tool_router]
impl ZebraMcpServer {
    #[tool(
        name = "fileTree",
        description = "List project files and directory structure. Use this instead of `find`, `ls -R`, or `glob` to discover source files — reads from the pre-built project index and returns instantly."
    )]
    async fn file_tree(
        &self,
        Parameters(params): Parameters<FileTreeParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let root_path = zrag_store::resolve_project(Some(&params.project))
            .await
            .map_err(|e| internal_err(format!("{e}")))?;
        let root = std::path::Path::new(&root_path);
        let pid = zrag_common::ids::project_id(root);
        let db = zrag_store::Db::open(&pid)
            .await
            .map_err(|e| internal_err(format!("store open: {e}")))?;
        let files = db
            .files_table()
            .await
            .map_err(|e| internal_err(format!("files_table: {e}")))?
            .list()
            .await
            .map_err(|e| internal_err(format!("list files: {e}")))?;

        let root_str = root.to_string_lossy();

        let matcher = params
            .path_glob
            .as_deref()
            .map(|p| {
                globset::Glob::new(p)
                    .map_err(|e| internal_err(format!("bad glob: {e}")))
                    .map(|g| g.compile_matcher())
            })
            .transpose()?;

        let mut matched = Vec::with_capacity(files.len());
        for f in &files {
            if match_file(&f.file_path, &root_str, matcher.as_ref()) {
                matched.push(f);
            }
        }

        let mut out = String::with_capacity(32 + matched.len() * 80);
        out.push_str("FILES\n");
        for (i, &f) in matched.iter().enumerate() {
            let rel = f
                .file_path
                .strip_prefix(root_str.as_ref())
                .unwrap_or(&f.file_path)
                .trim_start_matches('/');
            let _ = writeln!(out, "#{} [{}] {}", i, f.language, rel);
        }

        if matched.is_empty() {
            out.push_str("  (no files indexed)\n");
        }

        Ok(ok_text(out))
    }

    #[tool(
        name = "searchQuery",
        description = "Search the codebase by conceptual intent. Use this for broad exploration, feature discovery, or finding implementations based on natural language descriptions (e.g., 'database transaction rollback'). **Do not use this for exact symbol lookups or call-graph tracing.** If your query contains specific variables or exact class names, use `searchDep` instead. Returns complete source code chunks."
    )]
    async fn search_query(
        &self,
        Parameters(params): Parameters<SearchQueryParams>,
    ) -> Result<CallToolResult, ErrorData> {
        self.do_search(SearchCall {
            text: params.text,
            project: params.project.as_deref(),
            limit: params.limit,
            path_glob: params.path_glob,
            languages: params.languages,
            include_tests: params.include_tests,
            mode: SearchMode::Query,
        })
        .await
    }

    #[tool(
        name = "searchPassage",
        description = "Find code with similar structural semantics by example. Paste a block of code, a specific algorithmic pattern, or a complex error trace. Use this to locate duplicated logic, find other implementations of a specific interface, or hunt for similar anti-patterns across the entire project."
    )]
    async fn search_passage(
        &self,
        Parameters(params): Parameters<SearchPassageParams>,
    ) -> Result<CallToolResult, ErrorData> {
        self.do_search(SearchCall {
            text: params.text,
            project: params.project.as_deref(),
            limit: params.limit,
            path_glob: params.path_glob,
            languages: params.languages,
            include_tests: params.include_tests,
            mode: SearchMode::Passage,
        })
        .await
    }

    #[tool(
        name = "doctor",
        description = "DEBUG ONLY: Run diagnostics on the embedding engine and index. Use this ONLY when searchQuery or searchPassage return errors — not for empty results."
    )]
    async fn doctor(
        &self,
        Parameters(params): Parameters<DoctorParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let project_root = match &params.project {
            Some(p) => Some(
                zrag_store::resolve_project(Some(p))
                    .await
                    .map_err(|e| internal_err(format!("{e}")))?,
            ),
            None => None,
        };
        let req = DoctorReq { project_root };

        self.send(&Request::Doctor(req), |resp| match resp {
            Response::Doctor(Ok(report)) => {
                let mut out = String::with_capacity(256 + report.checks.len() * 64);
                let _ = writeln!(out, "Device: {}", report.device);
                for check in &report.checks {
                    let marker = match check.status {
                        CheckStatus::Ok => "OK",
                        CheckStatus::Warn => "WARN",
                        CheckStatus::Err => "ERR",
                    };
                    let _ = writeln!(out, "[{}] {}: {}", marker, check.name, check.message);
                }
                Ok(ok_text(out))
            }
            Response::Doctor(Err(e)) => Err(internal_err(e.message)),
            other => Err(internal_err(format!("unexpected: {other:?}"))),
        })
        .await
    }

    #[tool(
        name = "searchDep",
        description = "Look up an exact symbol to get its complete definition, location, and call graph (callers and callees up to a specified depth). **Use this as your primary tool when you know the exact name of a function, class, or variable.** Highly effective for tracing execution paths, data flow, and structural audits. Use fully qualified paths to resolve ambiguities."
    )]
    async fn search_dep(
        &self,
        Parameters(params): Parameters<SearchDepParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let project_root = zrag_store::resolve_project(params.project.as_deref())
            .await
            .map_err(|e| internal_err(format!("{e}")))?;
        let req = SearchDepReq {
            project_root,
            name: params.name,
            depth: params.depth,
            max_tokens: None,
        };
        let out = self.send_search_dep(req).await?;
        Ok(ok_text(out))
    }

    #[tool(
        name = "projectList",
        description = "Lists all indexed projects with root paths and stats. Call this when you need the `project` parameter for other tools and are unsure which project to target."
    )]
    async fn project_list(
        &self,
        Parameters(_): Parameters<ProjectListParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let projects = zrag_store::list_projects()
            .await
            .map_err(|e| internal_err(format!("list_projects: {e}")))?;

        if projects.is_empty() {
            return Ok(ok_text("No indexed projects found."));
        }

        Ok(ok_text(format_project_table(&projects)))
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for ZebraMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        let mut instructions = String::with_capacity(1024 + self.indexed_projects_roots.len());
        instructions.push_str(
            "# Semantic Code Search Server\n\
             \n\
             ## Tool Selection Guide (CRITICAL)\n\
             Choose your search tool based on the specific nature of your inquiry:\n\
             \n\
             1. **Exact Symbols & Call Graphs -> USE `searchDep`**\n\
                If you know the exact name of a function, class, type, or variable, or if you need to trace execution flow (who calls what), use this tool. This is your primary tool for structural and execution path audits.\n\
             \n\
             2. **Conceptual & Broad Discovery -> USE `searchQuery`**\n\
                If you are exploring abstract concepts (e.g., \"retry logic\", \"session validation\", \"fee calculation\"), use natural language here. \n\
                *Warning:* Do not use this for exact symbol lookups, as high-frequency keywords may dilute the search results.\n\
             \n\
             3. **Project Structure -> USE `fileTree`**\n\
                Use this to map the architecture of the project or find specific modules. This provides an instant, exact overview of the repository.\n\
             \n\
             4. **Pattern Matching -> USE `searchPassage`**\n\
                Use this to find code that behaves similarly to a specific snippet or error trace you have encountered.\n\
             \n\
             ## Managing Dependencies & Noise\n\
             External dependency directories are excluded at index time. Test files are indexed but hidden by default; set `include_tests: true` only when you explicitly need test code. Use `path_glob` to restrict searches to a specific source area when necessary.\n\
             \n\
             ## Pro Tips\n\
             * The `project` parameter auto-resolves if omitted. You can pass a name, index number, or root path.\n\
             * Search results contain the COMPLETE source code chunks. You do NOT need to call file-reading tools afterward to see the code.",
        );
        instructions.push_str(&self.indexed_projects_roots);
        info.instructions = Some(instructions);
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

pub fn run_mcp() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let indexed_projects_roots = match zrag_store::list_projects().await {
            Ok(projects) if projects.len() > 1 => {
                let mut s = String::with_capacity(32 + projects.len() * 80);
                s.push_str("\n\n## Indexed Projects\n\n");
                s.push_str(&format_project_table(&projects));
                s
            }
            _ => String::new(),
        };

        let server = ZebraMcpServer::new(indexed_projects_roots);
        let service = server.serve(stdio()).await?;
        service.waiting().await?;

        Ok(())
    })
}
