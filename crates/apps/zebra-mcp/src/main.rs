use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ErrorData, ServiceExt, tool};
use tokio::sync::{Mutex, RwLock};
use tracing_subscriber::EnvFilter;
use zti_common::format::format_elapsed;
use zti_dsl::{ProjectIndex, build_index, render::dsl::render_files_only};
use zti_ipc_client::Client;
use zti_protocol::format_search_results;
use zti_protocol::request::{DoctorReq, Request, SearchMode, SearchReq};
use zti_protocol::response::{CheckStatus, Response};

#[derive(Parser)]
#[command(
    name = "zebra-mcp",
    about = "Zebra MCP server (DSL + daemon IPC, stdio)"
)]
struct Cli;

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FileTreeParams {
    pub project_root: String,
    pub path_glob: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SearchParams {
    pub project_root: String,
    pub query: String,
    pub limit: Option<usize>,
    #[schemars(
        description = "When true, brute-force scan ALL embeddings instead of the fast approximate index. More accurate but significantly slower. Use only when approximate search misses relevant results."
    )]
    pub exhaustive: Option<bool>,
    #[schemars(
        description = "How the embedding model encodes the query: \"query\" (default, best for short keyword searches like 'find the auth handler') or \"passage\" (for longer descriptive input)."
    )]
    pub mode: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DoctorParams {
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
        description = "Maps the file structure. Use this to discover available source files and project roots in any project."
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
            "\n\n[SYSTEM HINT: Files discovered. Use `search` to find relevant code concepts.]",
        );
        Ok(ok_text(out))
    }

    #[tool(
        name = "search",
        description = "Finds relevant files or concepts. Returns paths."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mode = params
            .mode
            .as_deref()
            .map(|m| m.parse::<SearchMode>().unwrap_or_default())
            .unwrap_or_default();

        let req = SearchReq {
            project_root: params.project_root,
            query: params.query,
            limit: params.limit.unwrap_or(5),
            offset: None,
            languages: None,
            path_glob: None,
            refresh_index: false,
            exhaustive: params.exhaustive.unwrap_or(false),
            mode,
        };

        let mut guard = self.ensure_daemon().await?;
        let client = guard.as_mut().unwrap();

        match client.request(Request::Search(req)).await {
            Ok(Response::Search(Ok(results))) => {
                let mut out = format_search_results(&results);
                out.push_str("\n\n[SYSTEM HINT: Paths identified. Use `fileTree` to explore the file structure.]");
                Ok(ok_text(out))
            }
            Ok(Response::Search(Err(e))) => Err(internal_err(e.message)),
            Ok(other) => Err(internal_err(format!("unexpected: {other:?}"))),
            Err(e) => {
                *guard = None;
                Err(internal_err(format!("IPC lost, retry: {e}")))
            }
        }
    }

    #[tool(
        name = "doctor",
        description = "DEBUG ONLY: Run diagnostics on the embedding engine and database. Use this ONLY if the `search` tool returns system errors or fails to connect."
    )]
    async fn doctor(
        &self,
        Parameters(params): Parameters<DoctorParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let req = DoctorReq {
            project_root: params.project_root,
        };

        let mut guard = self.ensure_daemon().await?;
        let client = guard.as_mut().unwrap();

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
        description = "Lists all available indexed projects. Use this to discover project root paths before using other tools."
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

        out.push_str("\n\n[SYSTEM HINT: To explore a project, use `search` or `fileTree` with the project's root path.]");
        Ok(ok_text(out))
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for ZebraMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.instructions = Some(
            "CRITICAL: You are connected to the Zebra Tree Indexer. \
             DO NOT GUESS OR HALLUCINATE code paths or symbol identifiers. \
             Follow this strict workflow:\n\n\
             1. VERIFY: Use `projectList` to confirm the project is indexed.\n\
             2. EXPLORE: Use `fileTree` to map the workspace structure.\n\
             3. DISCOVER: Use `search` (semantic search) to find relevant \
             code by concept, function, or feature. This is the PRIMARY \
             discovery tool.\n\n\
             Rule: Never guess identifiers or code paths."
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
