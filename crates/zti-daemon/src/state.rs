use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, RwLock, watch};

use zti_common::ids;
use zti_dsl::ProjectIndex;
use zti_embed::EmbedEngine;
use zti_hw::Hardware;
use zti_store::Db;

pub struct LoadedProject {
    pub db: Db,
    pub dsl_index: RwLock<Option<Arc<ProjectIndex>>>,
    pub indexing_lock: Mutex<()>,
}

pub struct DaemonState {
    pub engine: Arc<EmbedEngine>,
    pub hardware: Hardware,
    pub registry: RwLock<HashMap<[u8; 32], Arc<LoadedProject>>>,
    pub started_at_ns: u64,
    pub started_at: Instant,
    pub shutdown_tx: watch::Sender<bool>,
    pub shutdown_rx: watch::Receiver<bool>,
    _pid_lock: File,
}

impl DaemonState {
    pub fn new(engine: EmbedEngine, hardware: Hardware, pid_lock: File) -> Self {
        let started_at_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        Self {
            engine: Arc::new(engine),
            hardware,
            registry: RwLock::new(HashMap::new()),
            started_at_ns,
            started_at: Instant::now(),
            shutdown_tx,
            shutdown_rx,
            _pid_lock: pid_lock,
        }
    }

    pub async fn load_or_open(&self, project_root: &str) -> anyhow::Result<Arc<LoadedProject>> {
        let root = std::path::Path::new(project_root).canonicalize()?;
        let pid = ids::project_id(&root);

        {
            let reg = self.registry.read().await;
            if let Some(proj) = reg.get(&pid) {
                return Ok(proj.clone());
            }
        }

        let db = Db::open(&pid).await?;
        let project = Arc::new(LoadedProject {
            db,
            dsl_index: RwLock::new(None),
            indexing_lock: Mutex::new(()),
        });

        {
            let mut reg = self.registry.write().await;
            reg.insert(pid, project.clone());
        }

        Ok(project)
    }
}
