use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use tokio::sync::Mutex;
use zti_ipc_client::Client;
use zti_protocol::response::SearchResults;
use zti_store::ProjectRow;

use super::registry::ModelEntry;

pub enum Screen {
    Setup(SetupPhase),
    Main,
}

impl Default for Screen {
    fn default() -> Self {
        Self::Setup(SetupPhase::default())
    }
}

#[derive(Default)]
pub enum SetupPhase {
    #[default]
    Resolving,
    FetchingRegistry,
    ModelSelection {
        entries: Arc<[ModelEntry]>,
        selected: usize,
    },
    DownloadingModel {
        model_id: Arc<str>,
    },
    VariantSelection {
        model_id: Arc<str>,
        variants: Vec<(Arc<str>, Arc<str>)>,
        selected: usize,
    },
    Launching {
        model_id: Arc<str>,
        variant: Arc<str>,
    },
    Error {
        message: String,
        can_retry: bool,
    },
}

#[derive(Default)]
pub enum DaemonStatus {
    #[default]
    Unknown,
    Starting,
    Running {
        model_id: String,
        device: String,
        uptime_secs: u64,
    },
    Stopped,
    Error(String),
}

#[derive(Default)]
pub enum ActivePanel {
    #[default]
    Projects,
    Search,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum DetailButton {
    #[default]
    Remove,
    Reindex,
    Back,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum AddConfirmButton {
    #[default]
    Confirm,
    Cancel,
}

pub enum Modal {
    ProjectDetail {
        selected_button: DetailButton,
    },
    ConfirmRemove,
    Error {
        message: String,
    },
    Indexing {
        current: u64,
        total: u64,
        message: String,
        is_reindex: bool,
    },
    AddProject {
        path_input: String,
        error: Option<String>,
    },
    AddProjectConfirm {
        canonical_path: String,
        already_indexed: bool,
        selected_button: AddConfirmButton,
    },
}

pub enum AppMessage {
    DaemonStatusUpdate(DaemonStatus),
    ProjectsLoaded(Vec<ProjectRow>),
    SearchDone(SearchResults),
    SearchError(String),
    ConfigResolved {
        model: Option<String>,
        variant: Option<String>,
    },
    RegistryLoaded(Vec<ModelEntry>),
    RegistryError(String),
    ModelDownloaded(Arc<str>),
    ModelDownloadError(String),
    SetupComplete {
        model: Arc<str>,
        variant: Arc<str>,
    },
    ProjectRemoved,
    ProjectRemoveError(String),
    IndexComplete,
    IndexProgress {
        current: u64,
        total: u64,
        message: String,
        is_reindex: bool,
    },
    IndexError(String),
}

pub struct App {
    pub screen: Screen,
    pub setup_registry: Option<Arc<[ModelEntry]>>,
    pub daemon_status: DaemonStatus,
    pub projects: Vec<ProjectRow>,
    pub selected_project: usize,
    pub active_panel: ActivePanel,
    pub modal: Option<Modal>,
    pub search_input: String,
    pub search_results: Option<SearchResults>,
    pub search_error: Option<String>,
    pub searching: bool,
    pub results_scroll: u16,
    pub should_quit: bool,
    pub client: Arc<Mutex<Option<Client>>>,
    pub model: Option<Arc<str>>,
    pub variant: Option<Arc<str>>,
    pub query_prefix: Option<Arc<str>>,
    pub passage_prefix: Option<Arc<str>>,
    pub should_run: Arc<AtomicBool>,
    pub monitor_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            screen: Screen::default(),
            setup_registry: None,
            daemon_status: DaemonStatus::default(),
            projects: Vec::with_capacity(8),
            selected_project: 0,
            active_panel: ActivePanel::default(),
            modal: None,
            search_input: String::with_capacity(256),
            search_results: None,
            search_error: None,
            searching: false,
            results_scroll: 0,
            should_quit: false,
            client: Arc::new(Mutex::new(None)),
            model: None,
            variant: None,
            query_prefix: None,
            passage_prefix: None,
            should_run: Arc::new(AtomicBool::new(true)),
            monitor_handle: None,
        }
    }
}

impl App {
    pub fn selected_project_root(&self) -> Option<&str> {
        self.projects
            .get(self.selected_project)
            .map(|p| p.root_path.as_str())
    }

    pub fn selected_project_root_owned(&self) -> Option<String> {
        self.projects
            .get(self.selected_project)
            .map(|p| p.root_path.clone())
    }

    pub fn apply_message(&mut self, msg: AppMessage) {
        match msg {
            AppMessage::DaemonStatusUpdate(status) => self.daemon_status = status,
            AppMessage::ProjectsLoaded(projects) => {
                self.projects = projects;
                let max = self.projects.len();
                if self.selected_project > max {
                    self.selected_project = max;
                }
            }
            AppMessage::SearchDone(results) => {
                self.search_results = Some(results);
                self.search_error = None;
                self.searching = false;
                self.results_scroll = 0;
            }
            AppMessage::SearchError(e) => {
                self.search_error = Some(e);
                self.searching = false;
            }
            _ => {}
        }
    }
}
