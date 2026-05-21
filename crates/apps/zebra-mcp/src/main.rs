use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ErrorData, ServiceExt, tool};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;
use zti_dsl::{
    AsciiTreeRenderer, ProjectIndex, build_index, render::dsl::render_files_only,
    render::dsl::DslRenderer,
};
use zti_tree_sitter::{parse_kinds, parse_language};

#[derive(Parser)]
#[command(name = "zebra-mcp", about = "Zebra MCP server (DSL-only, stdio)")]
struct Cli;

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct FileTreeParams {
    pub project_root: String,
    pub path_glob: Option<String>,
}

#[derive(Debug, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct ProjectMapParams {
    pub project_root: String,
    pub language: Option<String>,
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

#[derive(Debug, Clone)]
struct ZebraMcpServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    indexes: Arc<RwLock<HashMap<String, Arc<ProjectIndex>>>>,
}

impl ZebraMcpServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            indexes: Arc::new(RwLock::new(HashMap::with_capacity(4))),
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
}

fn ok_text(text: String) -> CallToolResult {
    CallToolResult::success(vec![Content::text(text)])
}

fn internal_err(msg: String) -> ErrorData {
    ErrorData::internal_error(msg, None)
}

#[rmcp::tool_router]
impl ZebraMcpServer {
    #[tool(description = "Returns the file tree with numeric #IDs")]
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

        Ok(ok_text(render_files_only(&index, &file_indices)))
    }

    #[tool(description = "Returns the DSL symbol map for a language")]
    async fn project_map(
        &self,
        Parameters(params): Parameters<ProjectMapParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let index = self.get_index(&params.project_root).await?;
        let max_tokens = params.max_tokens.unwrap_or(8000);

        let file_filter: Option<Vec<u16>> = params.language.as_ref().and_then(|l| {
            let lang = parse_language(l)?;
            Some(
                zti_dsl::files_by_language(&index.files, lang),
            )
        });

        let kind_filter = params.kinds.as_ref().map(|k| parse_kinds(k));

        let renderer = DslRenderer::new(&index, max_tokens);
        let text = renderer.render(file_filter.as_deref(), kind_filter.as_deref());

        Ok(ok_text(text))
    }

    #[tool(description = "Trace dependency chains for a symbol by its #ID")]
    async fn dep_tree(
        &self,
        Parameters(params): Parameters<DepTreeParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let index = self.get_index(&params.project_root).await?;
        let depth = params.depth.unwrap_or(2);

        let renderer = AsciiTreeRenderer::new(&index);
        let text = match params.direction.as_str() {
            "callers" => renderer.render_callers(params.symbol_id, depth),
            "callees" => renderer.render_callees(params.symbol_id, depth, false),
            other => {
                return Err(internal_err(format!(
                    "direction must be 'callers' or 'callees', got '{other}'"
                )));
            }
        };

        Ok(ok_text(text))
    }

    #[tool(description = "Read the exact source code of a symbol by its #ID")]
    async fn symbol_body(
        &self,
        Parameters(params): Parameters<SymbolBodyParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let index = self.get_index(&params.project_root).await?;

        let sym = index
            .symbols
            .get(params.symbol_id as usize)
            .ok_or_else(|| {
                internal_err(format!("Symbol {} not found", params.symbol_id))
            })?;
        let file = index
            .files
            .get(sym.file_idx as usize)
            .ok_or_else(|| {
                internal_err(format!("File for symbol {} not found", params.symbol_id))
            })?;

        let content = std::fs::read_to_string(&file.path)
            .map_err(|e| internal_err(format!("Failed to read {}: {e}", file.path)))?;

        let range = zti_common::line_byte_range(&content, sym.line, sym.end_line);
        let body = &content[range];
        let text = format!(
            "// File: {} | Lines: {}-{}\n{}",
            file.path, sym.line, sym.end_line, body
        );

        Ok(ok_text(text))
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for ZebraMcpServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.instructions = Some(
            "Zebra Tree Indexer MCP server (DSL-only). Use file_tree, project_map, dep_tree, \
              symbol_body for AST graph queries."
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
