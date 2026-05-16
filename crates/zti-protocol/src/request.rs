use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeReq {
    pub client_version: String,
    pub protocol_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexReq {
    pub project_root: String,
    pub refresh: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchReq {
    pub project_root: String,
    pub query: String,
    pub limit: usize,
    pub offset: Option<usize>,
    pub languages: Option<Vec<String>>,
    pub path_glob: Option<String>,
    pub refresh_index: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStatusReq {
    pub project_root: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveProjectReq {
    pub project_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReq {
    pub project_root: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTreeReq {
    pub project_root: String,
    pub path_glob: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMapReq {
    pub project_root: String,
    pub language: String,
    pub path_glob: Option<String>,
    pub kinds: Option<Vec<String>>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepTreeReq {
    pub project_root: String,
    pub symbol_id: u32,
    pub direction: String,
    pub depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolBodyReq {
    pub project_root: String,
    pub symbol_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    Handshake(HandshakeReq),
    Index(IndexReq),
    Search(SearchReq),
    ProjectStatus(ProjectStatusReq),
    DaemonStatus,
    RemoveProject(RemoveProjectReq),
    Stop,
    Doctor(DoctorReq),
    DaemonEnv,
    DslFileTree(FileTreeReq),
    DslProjectMap(ProjectMapReq),
    DslDepTree(DepTreeReq),
    DslSymbolBody(SymbolBodyReq),
}
