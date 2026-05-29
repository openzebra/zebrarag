use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    #[default]
    Query,
    Passage,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Query => "query",
            Self::Passage => "passage",
        })
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "query" => Ok(Self::Query),
            "passage" => Ok(Self::Passage),
            other => Err(format!("unknown search mode: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeReq {
    pub client_version: String,
    pub protocol_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexReq {
    pub project_root: String,
    pub refresh: bool,
    #[serde(default)]
    pub search_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelIndexReq {
    pub project_root: String,
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
    #[serde(default)]
    pub exhaustive: bool,
    #[serde(default)]
    pub mode: SearchMode,
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
    pub language: Option<String>,
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
pub struct SymbolBodiesReq {
    pub project_root: String,
    pub symbol_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    Handshake(HandshakeReq),
    Index(IndexReq),
    CancelIndex(CancelIndexReq),
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
    DslSymbolBodies(SymbolBodiesReq),
}
