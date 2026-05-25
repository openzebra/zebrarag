use std::sync::Arc;

use tokio::sync::Mutex;
use zti_ipc_client::Client;
use zti_protocol::response::SearchResults;
use zti_store::ProjectRow;

pub enum DaemonStatus {
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

pub enum ActivePanel {
    Projects,
    Search,
}

pub enum AppMessage {
    DaemonStatusUpdate(DaemonStatus),
    ProjectsLoaded(Vec<ProjectRow>),
    SearchDone(SearchResults),
    SearchError(String),
}

pub struct App {
    pub daemon_status: DaemonStatus,
    pub projects: Vec<ProjectRow>,
    pub selected_project: usize,
    pub active_panel: ActivePanel,
    pub search_input: String,
    pub search_results: Option<SearchResults>,
    pub search_error: Option<String>,
    pub searching: bool,
    pub results_scroll: u16,
    pub should_quit: bool,
    pub client: Arc<Mutex<Option<Client>>>,
}

impl App {
    pub fn new() -> Self {
        Self {
            daemon_status: DaemonStatus::Unknown,
            projects: Vec::new(),
            selected_project: 0,
            active_panel: ActivePanel::Projects,
            search_input: String::with_capacity(256),
            search_results: None,
            search_error: None,
            searching: false,
            results_scroll: 0,
            should_quit: false,
            client: Arc::new(Mutex::new(None)),
        }
    }

    pub fn selected_project_root(&self) -> Option<&str> {
        self.projects
            .get(self.selected_project)
            .map(|p| p.root_path.as_str())
    }

    pub fn apply_message(&mut self, msg: AppMessage) {
        match msg {
            AppMessage::DaemonStatusUpdate(status) => self.daemon_status = status,
            AppMessage::ProjectsLoaded(projects) => {
                self.projects = projects;
                if self.selected_project >= self.projects.len() && !self.projects.is_empty() {
                    self.selected_project = 0;
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
        }
    }
}
