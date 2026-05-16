use std::time::Duration;

use anyhow::Result;
use rmcp::ServiceExt;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::tool;
use rmcp::transport::stdio;
use tracing_subscriber::EnvFilter;

use zti_ipc_client::Client;
use zti_protocol::request::*;
use zti_protocol::response::*;

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
    tool_router: ToolRouter<Self>,
}

impl ZebraMcpServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    async fn client(&self) -> Result<Client> {
        let mut client = Client::connect(Duration::from_secs(10)).await?;
        client.handshake().await?;
        Ok(client)
    }
}

#[rmcp::tool_router]
impl ZebraMcpServer {
    #[tool(description = "Returns the file tree with numeric #IDs")]
    async fn file_tree(&self, Parameters(params): Parameters<FileTreeParams>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::DslFileTree(FileTreeReq {
                project_root: params.project_root,
                path_glob: params.path_glob,
            })).await?;
            match resp {
                Response::DslFileTree(Ok(body)) => Ok(body.text),
                Response::DslFileTree(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Returns the DSL symbol map for a language")]
    async fn project_map(&self, Parameters(params): Parameters<ProjectMapParams>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::DslProjectMap(ProjectMapReq {
                project_root: params.project_root,
                language: params.language,
                path_glob: params.path_glob,
                kinds: params.kinds,
                max_tokens: params.max_tokens,
            })).await?;
            match resp {
                Response::DslProjectMap(Ok(body)) => Ok(body.text),
                Response::DslProjectMap(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Trace dependency chains for a symbol")]
    async fn dep_tree(&self, Parameters(params): Parameters<DepTreeParams>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::DslDepTree(DepTreeReq {
                project_root: params.project_root,
                symbol_id: params.symbol_id,
                direction: params.direction,
                depth: params.depth,
            })).await?;
            match resp {
                Response::DslDepTree(Ok(body)) => Ok(body.text),
                Response::DslDepTree(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Read source code of a symbol")]
    async fn symbol_body(&self, Parameters(params): Parameters<SymbolBodyParams>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::DslSymbolBody(SymbolBodyReq {
                project_root: params.project_root,
                symbol_id: params.symbol_id,
            })).await?;
            match resp {
                Response::DslSymbolBody(Ok(body)) => Ok(body.text),
                Response::DslSymbolBody(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Search for code semantically")]
    async fn search(&self, Parameters(params): Parameters<SearchParams>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::Search(SearchReq {
                project_root: params.project_root,
                query: params.query,
                limit: params.limit.unwrap_or(5),
                offset: None,
                languages: None,
                path_glob: None,
                refresh_index: false,
            })).await?;
            match resp {
                Response::Search(Ok(results)) => {
                    let mut out = String::new();
                    for (i, hit) in results.hits.iter().enumerate() {
                        out.push_str(&format!("#{} {:.4} {} ({}:{}-{})\n",
                            i + 1, hit.score, hit.symbol_qualified,
                            hit.file_path, hit.start_line, hit.end_line));
                    }
                    Ok(out)
                }
                Response::Search(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Index a project")]
    async fn index(&self, Parameters(params): Parameters<IndexParams>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::Index(IndexReq {
                project_root: params.project_root,
                refresh: params.refresh.unwrap_or(false),
            })).await?;
            match resp {
                Response::Index(Ok(stats)) => Ok(format!(
                    "Indexed {} chunks in {} files ({:.1}s)",
                    stats.total_chunks, stats.total_files, stats.duration_ms as f64 / 1000.0
                )),
                Response::Index(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Show project status")]
    async fn project_status(&self, Parameters(params): Parameters<ProjectRootParam>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::ProjectStatus(ProjectStatusReq {
                project_root: params.project_root,
            })).await?;
            match resp {
                Response::ProjectStatus(Ok(s)) => Ok(format!(
                    "Root: {}\nModel: {} (dim={})\nChunks: {}\nFiles: {}",
                    s.project_root, s.model_id, s.model_dim, s.total_chunks, s.total_files
                )),
                Response::ProjectStatus(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Show daemon status")]
    async fn daemon_status(&self) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::DaemonStatus).await?;
            match resp {
                Response::DaemonStatus(s) => Ok(format!(
                    "Uptime: {}s\nProjects: {}\nModel: {}\nDevice: {}",
                    s.uptime_secs, s.projects_loaded, s.model_id, s.device
                )),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Remove a project")]
    async fn remove_project(&self, Parameters(params): Parameters<IndexParams>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::RemoveProject(RemoveProjectReq {
                project_root: params.project_root,
            })).await?;
            match resp {
                Response::RemoveProject(Ok(())) => Ok("Project removed.".to_string()),
                Response::RemoveProject(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Stop the daemon")]
    async fn stop(&self) -> String {
        async {
            let mut client = self.client().await?;
            let _ = client.request(Request::Stop).await?;
            Ok::<String, anyhow::Error>("Daemon stopped.".to_string())
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Run diagnostics")]
    async fn doctor(&self, Parameters(params): Parameters<ProjectRootParam>) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::Doctor(DoctorReq {
                project_root: params.project_root,
            })).await?;
            match resp {
                Response::Doctor(Ok(r)) => Ok(format!(
                    "Model: {} ({})\nDB: {} ({})\nDevice: {}",
                    r.model_path, if r.model_ok { "OK" } else { "MISSING" },
                    r.db_path, if r.db_ok { "OK" } else { "MISSING" },
                    r.device
                )),
                Response::Doctor(Err(e)) => Err(anyhow::anyhow!("{}", e.message)),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }

    #[tool(description = "Show daemon environment")]
    async fn daemon_env(&self) -> String {
        async {
            let mut client = self.client().await?;
            let resp = client.request(Request::DaemonEnv).await?;
            match resp {
                Response::DaemonEnv(env) => Ok(format!(
                    "Data: {}\nSocket: {}\nModel: {}\nDevice: {}\nCPUs: {}\nRAM: {}MB",
                    env.data_dir, env.socket_path, env.model_id, env.device, env.cpus, env.mem_total_mb
                )),
                _ => Err(anyhow::anyhow!("unexpected response")),
            }
        }.await.unwrap_or_else(|e| format!("Error: {}", e))
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for ZebraMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.instructions = Some("Zebra Tree Indexer MCP server. Use file_tree, project_map, dep_tree, symbol_body for DSL graph. Use search, index, project_status, daemon_status, remove_project, stop, doctor, daemon_env for daemon operations.".into());
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let server = ZebraMcpServer::new();
    let service = server.serve(stdio()).await?;
    service.waiting().await?;

    Ok(())
}
