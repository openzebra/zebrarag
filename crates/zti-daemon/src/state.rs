use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, RwLock, watch};

use zti_ann::AnnCache;
use zti_common::ids;
use zti_dsl::ProjectIndex;
use zti_embed::{EmbedEngine, LoadOverrides};
use zti_hw::Hardware;
use zti_store::Db;

pub struct LoadedProject {
    pub db: Db,
    pub dsl_index: RwLock<Option<Arc<ProjectIndex>>>,
    pub indexing_lock: Mutex<()>,
}

pub struct DaemonState {
    pub primary_model: Arc<str>,
    primary_engine: Arc<EmbedEngine>,
    pub engines: RwLock<HashMap<Arc<str>, Arc<EmbedEngine>>>,
    pub hardware: Hardware,
    pub registry: RwLock<HashMap<[u8; 32], Arc<LoadedProject>>>,
    pub ann: Arc<AnnCache>,
    pub started_at_ns: u64,
    pub started_at: Instant,
    pub shutdown_tx: watch::Sender<bool>,
    pub shutdown_rx: watch::Receiver<bool>,
    _pid_lock: File,
}

impl DaemonState {
    pub fn new(engine: EmbedEngine, model_id: Arc<str>, hardware: Hardware, pid_lock: File) -> Self {
        let started_at_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let primary = Arc::new(engine);
        let mut engines = HashMap::with_capacity(1);
        engines.insert(Arc::clone(&model_id), Arc::clone(&primary));

        Self {
            primary_model: model_id,
            primary_engine: primary,
            engines: RwLock::new(engines),
            hardware,
            registry: RwLock::new(HashMap::new()),
            ann: Arc::new(AnnCache::default()),
            started_at_ns,
            started_at: Instant::now(),
            shutdown_tx,
            shutdown_rx,
            _pid_lock: pid_lock,
        }
    }

    pub fn primary_engine(&self) -> Arc<EmbedEngine> {
        Arc::clone(&self.primary_engine)
    }

    pub async fn engine_for_model(&self, model_id: &str) -> anyhow::Result<Arc<EmbedEngine>> {
        {
            let engines = self.engines.read().await;
            if let Some(engine) = engines.get(model_id) {
                return Ok(Arc::clone(engine));
            }
        }

        let hw = self.hardware;
        let owned = model_id.to_owned();
        let engine = tokio::task::spawn_blocking(move || {
            EmbedEngine::load_with(&owned, &hw, &LoadOverrides::default())
        })
        .await??;

        let arc = Arc::new(engine);
        let mut engines = self.engines.write().await;
        engines.insert(Arc::from(model_id.to_owned()), Arc::clone(&arc));
        Ok(arc)
    }

    pub async fn load_or_open(&self, project_root: &str) -> anyhow::Result<Arc<LoadedProject>> {
        let root = std::path::Path::new(project_root).canonicalize()?;
        let pid = ids::project_id(&root);

        {
            let reg = self.registry.read().await;
            if let Some(proj) = reg.get(&pid) {
                return Ok(Arc::clone(proj));
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
            reg.insert(pid, Arc::clone(&project));
        }

        Ok(project)
    }
}
