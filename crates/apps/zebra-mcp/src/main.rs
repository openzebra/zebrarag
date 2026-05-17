use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{tool, ErrorData, ServiceExt};
use tracing_subscriber::EnvFilter;

use zti_ipc_client::Client;
use zti_protocol::request::*;
use zti_protocol::response::*;

#[derive(Parser)]
#[command(name = "zebra-mcp", about = "Zebra MCP server (stdio)")]
struct Cli {
    #[arg(short, long)]
    model: String,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct FileTreeParams {
    pub project_root: String,
    pub path_glob: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct ProjectMapParams {
    pub project_root: String,
    pub language: String,
    pub path_glob: Option<String>,
    pub kinds: Option<Vec<String>>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct DepTreeParams {
    pub project_root: String,
    pub symbol_id: u32,
    pub direction: String,
    pub depth: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct SymbolBodyParams {
    pub project_root: String,
    pub symbol_id: u32,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct SearchParams {
    pub project_root: String,
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct IndexParams {
    pub project_root: String,
    pub refresh: Option<bool>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct ProjectRootParam {
    pub project_root: Option<String>,
}

#[derive(Debug, Clone)]
struct ZebraMcpServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    model: String,
}

impl ZebraMcpServer {
    fn new(model: String) -> Self {
        Self {
            tool_router: Self::tool_router(),
            model,
        }
    }

    async fn client(&self) -> Result<Client, ErrorData> {
        let mut client = Client::connect(Duration::from_secs(10), &self.model)
            .await
            .map_err(daemon_err)?;
        client.handshake().await.map_err(daemon_err)?;
        Ok(client)
    }
}

/// Wrap a successful tool body text into a `CallToolResult`.
fn ok_text(text: String) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text)])
}

/// Daemon I/O failure (couldn't connect, framing error, etc.).
fn daemon_err(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(format!("daemon I/O: {}", e), None)
}

/// Daemon returned a structured `ErrorBody` — propagate as MCP error.
fn body_err(e: &ErrorBody) -> ErrorData {
    ErrorData::invalid_params(e.message.clone(), None)
}

/// Wrong `Response` variant returned — protocol bug.
fn unexpected_response() -> ErrorData {
    ErrorData::internal_error("daemon returned unexpected response variant", None)
}

#[rmcp::tool_router]
impl ZebraMcpServer {
    #[tool(description = "Returns the file tree with numeric #IDs")]
    async fn file_tree(
        &self,
        Parameters(params): Parameters<FileTreeParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::DslFileTree(FileTreeReq {
                project_root: params.project_root,
                path_glob: params.path_glob,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::DslFileTree(Ok(body)) => Ok(ok_text(body.text)),
            Response::DslFileTree(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Returns the DSL symbol map for a language")]
    async fn project_map(
        &self,
        Parameters(params): Parameters<ProjectMapParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::DslProjectMap(ProjectMapReq {
                project_root: params.project_root,
                language: params.language,
                path_glob: params.path_glob,
                kinds: params.kinds,
                max_tokens: params.max_tokens,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::DslProjectMap(Ok(body)) => Ok(ok_text(body.text)),
            Response::DslProjectMap(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Trace dependency chains for a symbol")]
    async fn dep_tree(
        &self,
        Parameters(params): Parameters<DepTreeParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::DslDepTree(DepTreeReq {
                project_root: params.project_root,
                symbol_id: params.symbol_id,
                direction: params.direction,
                depth: params.depth,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::DslDepTree(Ok(body)) => Ok(ok_text(body.text)),
            Response::DslDepTree(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Read source code of a symbol")]
    async fn symbol_body(
        &self,
        Parameters(params): Parameters<SymbolBodyParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::DslSymbolBody(SymbolBodyReq {
                project_root: params.project_root,
                symbol_id: params.symbol_id,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::DslSymbolBody(Ok(body)) => Ok(ok_text(body.text)),
            Response::DslSymbolBody(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Search for code semantically")]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::Search(SearchReq {
                project_root: params.project_root,
                query: params.query,
                limit: params.limit.unwrap_or(5),
                offset: None,
                languages: None,
                path_glob: None,
                refresh_index: false,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::Search(Ok(results)) => {
                let mut out = String::new();
                use std::fmt::Write as _;
                for (i, hit) in results.hits.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "#{} {:.4} {} ({}:{}-{})",
                        i + 1,
                        hit.score,
                        hit.symbol_qualified,
                        hit.file_path,
                        hit.start_line,
                        hit.end_line
                    );
                }
                Ok(ok_text(out))
            }
            Response::Search(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Index a project")]
    async fn index(
        &self,
        Parameters(params): Parameters<IndexParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        // Progress frames are drained into a counter; we report the totals
        // alongside the terminal stats. Future work (N11 follow-up): forward
        // these to the MCP client via `Peer::notify_progress` when the caller
        // supplied a `progressToken` in the request meta.
        let progress_count = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&progress_count);
        let resp = client
            .request_streaming(
                Request::Index(IndexReq {
                    project_root: params.project_root,
                    refresh: params.refresh.unwrap_or(false),
                }),
                move |frame| {
                    if let Response::IndexProgress(_) = frame {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                },
            )
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::Index(Ok(stats)) => Ok(ok_text(format!(
                "Indexed {} chunks in {} files ({:.1}s, {} progress frames)",
                stats.total_chunks,
                stats.total_files,
                stats.duration_ms as f64 / 1000.0,
                progress_count.load(Ordering::Relaxed),
            ))),
            Response::Index(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Show project status")]
    async fn project_status(
        &self,
        Parameters(params): Parameters<ProjectRootParam>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::ProjectStatus(ProjectStatusReq {
                project_root: params.project_root,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::ProjectStatus(Ok(s)) => Ok(ok_text(format!(
                "Root: {}\nModel: {} (dim={})\nChunks: {}\nFiles: {}",
                s.project_root, s.model_id, s.model_dim, s.total_chunks, s.total_files
            ))),
            Response::ProjectStatus(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Show daemon status")]
    async fn daemon_status(&self) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::DaemonStatus)
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::DaemonStatus(s) => Ok(ok_text(format!(
                "Uptime: {}s\nProjects: {}\nModel: {}\nDevice: {}",
                s.uptime_secs, s.projects_loaded, s.model_id, s.device
            ))),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Remove a project")]
    async fn remove_project(
        &self,
        Parameters(params): Parameters<IndexParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::RemoveProject(RemoveProjectReq {
                project_root: params.project_root,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::RemoveProject(Ok(())) => Ok(ok_text("Project removed.".to_string())),
            Response::RemoveProject(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Stop the daemon")]
    async fn stop(&self) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let _ = client
            .request(Request::Stop)
            .await
            .map_err(daemon_err)?;
        Ok(ok_text("Daemon stopped.".to_string()))
    }

    #[tool(description = "Run diagnostics")]
    async fn doctor(
        &self,
        Parameters(params): Parameters<ProjectRootParam>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::Doctor(DoctorReq {
                project_root: params.project_root,
            }))
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::Doctor(Ok(r)) => {
                use std::fmt::Write as _;
                let mut out = String::new();
                let _ = writeln!(out, "Device: {}", r.device);
                for check in &r.checks {
                    let marker = match check.status {
                        CheckStatus::Ok => "OK",
                        CheckStatus::Warn => "WARN",
                        CheckStatus::Err => "ERR",
                    };
                    let _ = writeln!(out, "[{}] {}: {}", marker, check.name, check.message);
                }
                Ok(ok_text(out))
            }
            Response::Doctor(Err(e)) => Err(body_err(&e)),
            _ => Err(unexpected_response()),
        }
    }

    #[tool(description = "Show daemon environment")]
    async fn daemon_env(&self) -> Result<CallToolResult, ErrorData> {
        let mut client = self.client().await?;
        let resp = client
            .request(Request::DaemonEnv)
            .await
            .map_err(daemon_err)?;
        match resp {
            Response::DaemonEnv(env) => Ok(ok_text(format!(
                "Data: {}\nSocket: {}\nModel: {}\nDevice: {}\nCPUs: {}\nRAM: {}MB",
                env.data_dir,
                env.socket_path,
                env.model_id,
                env.device,
                env.cpus,
                env.mem_total_mb
            ))),
            _ => Err(unexpected_response()),
        }
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for ZebraMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.instructions = Some(
            "Zebra Tree Indexer MCP server. Use file_tree, project_map, dep_tree, \
             symbol_body for DSL graph. Use search, index, project_status, \
             daemon_status, remove_project, stop, doctor, daemon_env for daemon \
             operations."
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

    let Cli { model } = Cli::parse();
    let server = ZebraMcpServer::new(model);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
