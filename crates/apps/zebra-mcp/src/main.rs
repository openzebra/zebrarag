use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{tool, ErrorData, ServiceExt};
use tokio::sync::{Mutex, RwLock};
use tracing_subscriber::EnvFilter;
use zti_common::format::format_elapsed;
use zti_dsl::{build_index, render::dsl::render_files_only, ProjectIndex};
use zti_ipc_client::Client;
use zti_protocol::format_search_results;
use zti_protocol::request::{DoctorReq, Request, SearchMode, SearchReq};
use zti_protocol::response::{CheckStatus, Response, SearchResults};

#[derive(Parser)]
#[command(
    name = "zebra-mcp",
    about = "Zebra MCP server (DSL + daemon IPC, stdio)"
)]
struct Cli;

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FileTreeParams {
    #[schemars(description = "Absolute path to the project root. Obtain valid paths from `projectList`.")]
    pub project_root: String,
    #[schemars(description = "Optional glob pattern to filter files, e.g. \"**/*.rs\" or \"src/**/*.ts\".")]
    pub path_glob: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct SearchQueryParams {
    #[schemars(description = "Natural language query. Use full phrases: \"user authentication middleware\" not \"auth\".")]
    pub text: String,
    #[schemars(description = "Project root path. Auto-resolved when omitted if only one project is indexed.")]
    pub root: Option<String>,
    #[schemars(description = "Maximum results to return (default: 5).")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct SearchPassageParams {
    #[schemars(description = "A code snippet, error log, or descriptive paragraph to find similar implementations.")]
    pub text: String,
    #[schemars(description = "Project root path. Auto-resolved when omitted if only one project is indexed.")]
    pub root: Option<String>,
    #[schemars(description = "Maximum results to return (default: 5).")]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DoctorParams {
    #[schemars(description = "Optional project root to diagnose. If omitted, runs general diagnostics.")]
    pub project_root: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProjectListParams {}

#[derive(Clone)]
struct ZebraMcpServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    indexes: Arc<RwLock<HashMap<String, Arc<ProjectIndex>>>>,
    daemon: Arc<Mutex<Option<Client>>>,
}

impl ZebraMcpServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            indexes: Arc::new(RwLock::new(HashMap::with_capacity(4))),
            daemon: Arc::new(Mutex::new(None)),
        }
    }

    async fn get_index(&self, project_root: &str) -> Result<Arc<ProjectIndex>, ErrorData> {
        let root = std::path::Path::new(project_root)
            .canonicalize()
            .map_err(|e| internal_err(format!("invalid project_root: {e}")))?;
        let root_key = root.to_string_lossy().to_string();

        let mut guard = self.indexes.write().await;
        match guard.entry(root_key) {
            Entry::Occupied(e) => Ok(Arc::clone(e.get())),
            Entry::Vacant(e) => {
                let key = e.key().clone();
                let idx = Arc::new(
                    tokio::task::spawn_blocking(move || build_index(&key))
                        .await
                        .map_err(|e| internal_err(format!("indexing task failed: {e}")))?
                        .map_err(|e| internal_err(format!("indexing failed: {e}")))?,
                );
                e.insert(Arc::clone(&idx));
                Ok(idx)
            }
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
            out.push_str(
                "\n\n[SYSTEM HINT: Use `fileTree` to explore the project file structure.]",
            );
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
            out.push_str(
                "\n\n[SYSTEM HINT: No results found. Try rephrasing with more descriptive terms.]",
            );
        } else {
            out.push_str(
                "\n\n[SYSTEM HINT: Use `fileTree` to explore the project file structure.]",
            );
        }
        Ok(ok_text(out))
    }
}

fn ok_text(text: String) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text)])
}

fn internal_err(msg: String) -> ErrorData {
    ErrorData::internal_error(msg, None)
}

#[rmcp::tool_router]
impl ZebraMcpServer {
    #[tool(
        name = "fileTree",
        description = "Maps the file structure. Use this to discover available source files and project roots in any project. Prefer this over `glob` or `find` — it uses the indexed project tree."
    )]
    async fn file_tree(
        &self,
        Parameters(params): Parameters<FileTreeParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let index = self.get_index(&params.project_root).await?;

        let file_indices: Vec<u16> = if let Some(glob) = &params.path_glob {
            zti_dsl::glob_match_files(&index.files, &index.root, glob)
                .map_err(|e| internal_err(format!("bad glob: {e}")))?
        } else {
            (0..index.files.len() as u16).collect()
        };

        let mut out = render_files_only(&index, &file_indices);
        out.push_str(
            "\n\n[SYSTEM HINT: Files discovered. Use `searchQuery` to find code concepts or `searchPassage` to find similar code.]",
        );
        Ok(ok_text(out))
    }

    #[tool(
        name = "searchQuery",
        description = "Semantic search with natural language. Returns pre-chunked code snippets with file paths and line ranges — more token-efficient than reading entire files. Understands intent: \"authentication middleware\" finds auth handlers even without that exact string."
    )]
    async fn search_query(
        &self,
        Parameters(params): Parameters<SearchQueryParams>,
    ) -> Result<CallToolResult, ErrorData> {
        self.do_search(params.text, params.root.as_deref(), params.limit, SearchMode::Query)
            .await
    }

    #[tool(
        name = "searchPassage",
        description = "Find similar code by example. Paste a code snippet, error message, or describe an implementation pattern to locate related code across the project."
    )]
    async fn search_passage(
        &self,
        Parameters(params): Parameters<SearchPassageParams>,
    ) -> Result<CallToolResult, ErrorData> {
        self.do_search(params.text, params.root.as_deref(), params.limit, SearchMode::Passage)
            .await
    }

    #[tool(
        name = "doctor",
        description = "DEBUG ONLY: Run diagnostics on the embedding engine and database. Use this ONLY if searchQuery or searchPassage return system errors."
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
        description = "Lists all indexed projects with root paths. Useful when multiple projects are indexed and you need to pick the right `root`."
    )]
    async fn project_list(
        &self,
        Parameters(_): Parameters<ProjectListParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let projects = zti_store::list_projects()
            .await
            .map_err(|e| internal_err(format!("list_projects: {e}")))?;

        if projects.is_empty() {
            return Ok(ok_text(String::from("No indexed projects found.")));
        }

        let mut out = String::with_capacity(projects.len() * 128);
        out.push_str("| Project | Model | Chunks | Files | Last Indexed |\n");
        out.push_str("|---------|-------|--------|-------|-------------|\n");
        for p in &projects {
            let name = std::path::Path::new(&p.root_path)
                .file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_else(|| Cow::Borrowed(&p.root_path));
            let ago = format_elapsed(p.last_indexed_ns);
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} |",
                name, p.model_id, p.total_chunks, p.total_files, ago
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
        info.instructions = Some(
            "# zebra-mcp — Semantic Code Search\n\
             \n\
             Semantic search over your codebase using vector embeddings. \
             Returns pre-chunked code snippets with file paths and line \
             ranges — more token-efficient than reading entire files.\n\
             \n\
             ## Tools\n\
             \n\
             * **`searchQuery`** — Natural language search. Ask a question \
             or describe what you're looking for: \"database connection pool\", \
             \"error retry logic\", \"the function that parses CLI arguments\".\n\
             \n\
             * **`searchPassage`** — Similarity search. Paste a code snippet \
             or describe a pattern to find related implementations.\n\
             \n\
             * **`fileTree`** — Browse the indexed project file structure.\n\
             \n\
             * **`projectList`** — List all indexed projects with root paths.\n\
             \n\
             * **`doctor`** — Debug connectivity and index health \
             (use only when tools return errors).\n\
             \n\
             ## Tips\n\
             \n\
             * Use descriptive phrases, not single keywords. \
             \"user session validation\" finds more than \"auth\".\n\
             * The `root` parameter is optional when only one project \
             is indexed — it auto-resolves.\n\
             * Results include line ranges — read just the relevant \
             section instead of whole files.\n\
             * If the fast index misses results, exhaustive search \
             runs automatically. No manual retry needed."
                .into(),
        );
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let _cli = Cli::parse();
    let server = ZebraMcpServer::new();
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
