use std::borrow::Cow;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ErrorData, ServiceExt, tool};
use tokio::sync::Mutex;
use zti_common::format::format_elapsed;
use zti_ipc_client::Client;
use zti_protocol::format_search_results;
use zti_protocol::request::{DoctorReq, Request, SearchMode, SearchReq};
use zti_protocol::response::{CheckStatus, Response, SearchResults};

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FileTreeParams {
    #[schemars(
        description = "Absolute path to the project root. Obtain valid paths from `projectList`."
    )]
    pub project_root: String,
    #[schemars(
        description = "Optional glob pattern to filter files, e.g. \"**/*.rs\" or \"src/**/*.ts\"."
    )]
    pub path_glob: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct SearchQueryParams {
    #[schemars(
        description = "What you're looking for, in natural language. Use descriptive phrases: \"polynomial inversion\" not \"invert\"."
    )]
    pub text: String,
    #[schemars(
        description = "Project root path. Auto-resolved when omitted if only one project is indexed."
    )]
    pub root: Option<String>,
    #[schemars(description = "Maximum results to return (default: 5).")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct SearchPassageParams {
    #[schemars(
        description = "A code snippet, error message, or descriptive paragraph to find similar implementations."
    )]
    pub text: String,
    #[schemars(
        description = "Project root path. Auto-resolved when omitted if only one project is indexed."
    )]
    pub root: Option<String>,
    #[schemars(description = "Maximum results to return (default: 5).")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DoctorParams {
    #[schemars(
        description = "Optional project root to diagnose. If omitted, runs general diagnostics."
    )]
    pub project_root: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectListParams {}

#[derive(Clone)]
struct ZebraMcpServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    daemon: Arc<Mutex<Option<Client>>>,
    indexed_projects_roots: String,
}

impl ZebraMcpServer {
    fn new(indexed_projects_roots: String) -> Self {
        Self {
            tool_router: Self::tool_router(),
            daemon: Arc::new(Mutex::new(None)),
            indexed_projects_roots,
        }
    }

    async fn ensure_daemon(
        &self,
    ) -> Result<tokio::sync::MutexGuard<'_, Option<Client>>, ErrorData> {
        let mut guard = self.daemon.lock().await;
        if guard.is_none() {
            let mut client = Client::connect(Duration::from_secs(10), None, None, None, None)
                .await
                .map_err(|e| internal_err(format!("daemon connect: {e}")))?;
            client
                .handshake()
                .await
                .map_err(|e| internal_err(format!("handshake: {e}")))?;
            *guard = Some(client);
        }
        Ok(guard)
    }

    async fn resolve_project_root(root: Option<&str>) -> Result<String, ErrorData> {
        match root {
            Some(r) => {
                let canonical = std::path::Path::new(r)
                    .canonicalize()
                    .map_err(|e| internal_err(format!("invalid root path: {e}")))?;
                Ok(canonical.to_string_lossy().into_owned())
            }
            None => {
                let projects = zti_store::list_projects()
                    .await
                    .map_err(|e| internal_err(format!("list_projects: {e}")))?;

                match projects.len() {
                    0 => Err(internal_err(
                        "No indexed projects. Index a project first.".into(),
                    )),
                    1 => projects
                        .into_iter()
                        .next()
                        .map(|p| p.root_path)
                        .ok_or_else(|| internal_err("empty project list".into())),
                    _ => {
                        let mut msg = String::with_capacity(64 + projects.len() * 80);
                        msg.push_str("Multiple projects indexed. Specify `root`:\n");
                        for p in &projects {
                            let name = std::path::Path::new(&p.root_path)
                                .file_name()
                                .map(|s| s.to_string_lossy())
                                .unwrap_or(Cow::Borrowed(&p.root_path));
                            let _ = writeln!(msg, "  - {} ({})", name, p.root_path);
                        }
                        Err(internal_err(msg))
                    }
                }
            }
        }
    }

    async fn send_search(&self, req: SearchReq) -> Result<SearchResults, ErrorData> {
        let mut guard = self.ensure_daemon().await?;
        let client = guard
            .as_mut()
            .ok_or_else(|| internal_err("daemon not initialized".into()))?;

        match client.request(Request::Search(req)).await {
            Ok(Response::Search(Ok(results))) => Ok(results),
            Ok(Response::Search(Err(e))) => Err(internal_err(e.message)),
            Ok(other) => Err(internal_err(format!("unexpected response: {other:?}"))),
            Err(e) => {
                *guard = None;
                Err(internal_err(format!("IPC lost, retry: {e}")))
            }
        }
    }

    async fn do_search(
        &self,
        text: String,
        root: Option<&str>,
        limit: Option<usize>,
        mode: SearchMode,
    ) -> Result<CallToolResult, ErrorData> {
        let project_root = Self::resolve_project_root(root).await?;
        let limit = limit.unwrap_or(5);

        let req = SearchReq {
            project_root: project_root.clone(),
            query: text.clone(),
            limit,
            offset: None,
            languages: None,
            path_glob: None,
            refresh_index: false,
            exhaustive: false,
            mode,
        };

        let results = self.send_search(req).await?;

        if !results.hits.is_empty() {
            let mut out = format_search_results(&results);
            out.push_str(HINT_CODE_IN_CONTEXT);
            return Ok(ok_text(out));
        }

        let retry_req = SearchReq {
            project_root,
            query: text,
            limit,
            offset: None,
            languages: None,
            path_glob: None,
            refresh_index: false,
            exhaustive: true,
            mode,
        };

        let retry_results = self.send_search(retry_req).await?;
        let mut out = format_search_results(&retry_results);

        if retry_results.hits.is_empty() {
            out.push_str(HINT_NO_RESULTS);
        } else {
            out.push_str(HINT_CODE_IN_CONTEXT);
        }
        Ok(ok_text(out))
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

const HINT_CODE_IN_CONTEXT: &str = "\n\n[SYSTEM HINT: The source code above is already in your context. \
     Do NOT re-read these files — use the code directly. \
     For other files, use `searchQuery` or `fileTree`.]";

const HINT_NO_RESULTS: &str =
    "\n\n[SYSTEM HINT: No results found. Try rephrasing with more descriptive terms.]";

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
        let root = std::path::Path::new(&params.project_root)
            .canonicalize()
            .map_err(|e| internal_err(format!("invalid project_root: {e}")))?;
        let pid = zti_common::ids::project_id(&root);
        let db = zti_store::Db::open(&pid)
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

        let matched: Vec<&zti_store::FileRow> = files
            .iter()
            .filter(|f| match_file(&f.file_path, &root_str, matcher.as_ref()))
            .collect();

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

        out.push_str(
            "\n\n[SYSTEM HINT: Files discovered. Use `searchQuery` to find code concepts \
             or `searchPassage` to find similar code.]",
        );
        Ok(ok_text(out))
    }

    #[tool(
        name = "searchQuery",
        description = "Search the codebase by intent. Use this FIRST when exploring code, answering questions about the codebase, or finding implementations — before grep, find, or reading files. Describe what you need in plain language (e.g. \"polynomial inversion\", \"error retry logic\"). Returns complete source code with file paths and line ranges — no follow-up file reads needed."
    )]
    async fn search_query(
        &self,
        Parameters(params): Parameters<SearchQueryParams>,
    ) -> Result<CallToolResult, ErrorData> {
        self.do_search(
            params.text,
            params.root.as_deref(),
            params.limit,
            SearchMode::Query,
        )
        .await
    }

    #[tool(
        name = "searchPassage",
        description = "Find similar code by example. Paste a code snippet, error message, or pattern description to locate related implementations. Use this instead of grepping for exact matches when you want semantically similar code."
    )]
    async fn search_passage(
        &self,
        Parameters(params): Parameters<SearchPassageParams>,
    ) -> Result<CallToolResult, ErrorData> {
        self.do_search(
            params.text,
            params.root.as_deref(),
            params.limit,
            SearchMode::Passage,
        )
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
        let req = DoctorReq {
            project_root: params.project_root,
        };

        let mut guard = self.ensure_daemon().await?;
        let client = guard
            .as_mut()
            .ok_or_else(|| internal_err("daemon not initialized".into()))?;

        match client.request(Request::Doctor(req)).await {
            Ok(Response::Doctor(Ok(report))) => {
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
            Ok(Response::Doctor(Err(e))) => Err(internal_err(e.message)),
            Ok(other) => Err(internal_err(format!("unexpected: {other:?}"))),
            Err(e) => {
                *guard = None;
                Err(internal_err(format!("IPC lost, retry: {e}")))
            }
        }
    }

    #[tool(
        name = "projectList",
        description = "Lists all indexed projects with root paths and stats. Call this when you need the `root` parameter for other tools and are unsure which project to target."
    )]
    async fn project_list(
        &self,
        Parameters(_): Parameters<ProjectListParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let projects = zti_store::list_projects()
            .await
            .map_err(|e| internal_err(format!("list_projects: {e}")))?;

        if projects.is_empty() {
            return Ok(ok_text("No indexed projects found."));
        }

        let mut out = String::with_capacity(projects.len() * 128);
        out.push_str("| Project | Root | Model | Chunks | Files | Last Indexed |\n");
        out.push_str("|---------|------|-------|--------|-------|-------------|\n");
        for p in &projects {
            let name = std::path::Path::new(&p.root_path)
                .file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_else(|| Cow::Borrowed(&p.root_path));
            let ago = format_elapsed(p.last_indexed_ns);
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} |",
                name, p.root_path, p.model_id, p.total_chunks, p.total_files, ago
            );
        }

        out.push_str("\n\n[SYSTEM HINT: To explore a project, use `searchQuery`, `searchPassage`, or `fileTree`. The `root` parameter is optional when only one project is indexed.]");
        Ok(ok_text(out))
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for ZebraMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        let mut instructions = String::with_capacity(1024 + self.indexed_projects_roots.len());
        instructions.push_str(
            "# zebraindex — Semantic Code Search\n\
             \n\
             ## When to use these tools\n\
             \n\
             Use `searchQuery` as your **first step** when exploring code, answering \
             questions, or locating implementations. It replaces grep, find, and \
             manual file browsing — it understands what you mean, not just what you \
             type, and returns complete source code in a single call.\n\
             \n\
             ## Workflow\n\
             \n\
             1. **Start with `searchQuery`** — describe what you're looking for \
             in natural language. Results include the full source code with \
             file paths and line ranges. No second read step needed.\n\
             \n\
             2. **Use `searchPassage`** when you have a code snippet or error \
             message and want to find similar patterns across the project.\n\
             \n\
             3. **Use `fileTree`** to discover project structure — prefer it \
             over `find` or `ls`.\n\
             \n\
             ## Tips\n\
             \n\
             * Use descriptive phrases, not single keywords. \
             \"user session validation\" finds more than \"auth\".\n\
             * The `root` parameter auto-resolves when only one project is indexed.\n\
             * Results contain complete source code — use it directly without \
             re-reading files.\n\
             * If the fast index misses results, exhaustive search runs \
             automatically.",
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
        let indexed_projects_roots = match zti_store::list_projects().await {
            Ok(projects) if projects.len() > 1 => {
                let mut s = String::with_capacity(32 + projects.len() * 64);
                s.push_str("\n\n## Indexed Projects\n");
                for p in &projects {
                    let _ = writeln!(s, "- {}", p.root_path);
                }
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
