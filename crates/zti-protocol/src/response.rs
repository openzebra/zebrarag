use std::borrow::Cow;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResp {
    pub ok: bool,
    pub daemon_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_chunks: usize,
    pub total_files: usize,
    pub new_chunks: usize,
    pub reindexed_files: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingProgress {
    pub phase: String,
    pub current: u64,
    pub total: u64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk_id: Vec<u8>,
    pub file_path: String,
    pub symbol_qualified: String,
    pub symbol_kind: String,
    pub sym_id: u32,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub hits: Vec<SearchHit>,
    pub appendix: Vec<SearchHit>,
    pub legend: Cow<'static, str>,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStatus {
    pub project_root: String,
    pub total_chunks: u64,
    pub total_files: u64,
    pub model_id: String,
    pub model_dim: u32,
    pub last_indexed_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatusInfo {
    pub started_at_ns: u64,
    pub uptime_secs: u64,
    pub projects_loaded: usize,
    pub model_id: String,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus {
    Ok,
    Warn,
    Err,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorCheck {
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub device: String,
    pub checks: Vec<DoctorCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonEnvInfo {
    pub data_dir: String,
    pub socket_path: String,
    pub model_id: String,
    pub device: String,
    pub cpus: u32,
    pub mem_total_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTreeBody {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMapBody {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepTreeBody {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolBodyBody {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    Handshake(HandshakeResp),
    Index(Result<IndexStats, ErrorBody>),
    IndexProgress(IndexingProgress),
    Search(Result<SearchResults, ErrorBody>),
    ProjectStatus(Result<ProjectStatus, ErrorBody>),
    DaemonStatus(DaemonStatusInfo),
    RemoveProject(Result<(), ErrorBody>),
    Stop(()),
    Doctor(Result<DoctorReport, ErrorBody>),
    DaemonEnv(DaemonEnvInfo),
    DslFileTree(Result<FileTreeBody, ErrorBody>),
    DslProjectMap(Result<ProjectMapBody, ErrorBody>),
    DslDepTree(Result<DepTreeBody, ErrorBody>),
    DslSymbolBody(Result<SymbolBodyBody, ErrorBody>),
}
