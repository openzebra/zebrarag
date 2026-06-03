use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use tokio::sync::{Mutex, RwLock, watch};

use zti_ann::AnnCache;
use zti_common::ids;
use zti_dsl::ProjectIndex;
use zti_embed::{EmbedEngine, LoadOverrides};
use zti_rerank::gpu::TurboScorerCache;
use zti_store::Db;

pub struct LoadedProject {
    pub db: Db,
    pub dsl_index: RwLock<Option<Arc<ProjectIndex>>>,
    pub indexing_lock: Mutex<()>,
    pub cancel: AtomicBool,
    /// Set while an auto-reindex is queued or running, so overlapping FS events
    /// coalesce into one trailing run instead of stacking tasks.
    pub reindex_scheduled: AtomicBool,
}

pub struct DaemonState {
    pub primary_model: Arc<str>,
    primary_engine: Arc<EmbedEngine>,
    pub engines: RwLock<HashMap<Arc<str>, Arc<EmbedEngine>>>,
    pub loading_model: RwLock<Option<Arc<str>>>,
    pub hardware: Arc<zti_hw::Hardware>,
    pub model_dtype: Option<String>,
    pub registry: RwLock<HashMap<[u8; 32], Arc<LoadedProject>>>,
    pub ann: Arc<AnnCache>,
    pub turbo: Arc<TurboScorerCache>,
    pub started_at_ns: u64,
    pub started_at: Instant,
    pub shutdown_tx: watch::Sender<bool>,
    pub shutdown_rx: watch::Receiver<bool>,
    pub watch: OnceLock<Arc<crate::watch::WatchManager>>,
    _pid_lock: File,
}

impl DaemonState {
    pub fn new(
        engine: EmbedEngine,
        model_id: Arc<str>,
        hardware: Arc<zti_hw::Hardware>,
        model_dtype: Option<String>,
        pid_lock: File,
    ) -> Self {
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
            loading_model: RwLock::new(None),
            hardware,
            model_dtype,
            registry: RwLock::new(HashMap::with_capacity(4)),
            ann: Arc::new(AnnCache::default()),
            turbo: Arc::new(TurboScorerCache::default()),
            started_at_ns,
            started_at: Instant::now(),
            shutdown_tx,
            shutdown_rx,
            watch: OnceLock::new(),
            _pid_lock: pid_lock,
        }
    }

    pub fn primary_engine(&self) -> Arc<EmbedEngine> {
        Arc::clone(&self.primary_engine)
    }

    pub fn device_str(&self) -> &str {
        self.hardware.device.as_str()
    }

    pub async fn engine_for_model(&self, model_id: &str) -> anyhow::Result<Arc<EmbedEngine>> {
        {
            let engines = self.engines.read().await;
            if let Some(engine) = engines.get(model_id) {
                return Ok(Arc::clone(engine));
            }
        }

        {
            *self.loading_model.write().await = Some(Arc::from(model_id));
        }

        let hw = Arc::clone(&self.hardware);
        let owned = model_id.to_owned();
        let result = tokio::task::spawn_blocking(move || {
            EmbedEngine::load_with(&owned, hw, &LoadOverrides::default())
        })
        .await?;

        {
            *self.loading_model.write().await = None;
        }

        let engine = result?;
        let arc = Arc::new(engine);
        let mut engines = self.engines.write().await;
        engines.insert(Arc::from(model_id), Arc::clone(&arc));
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
            cancel: AtomicBool::new(false),
            reindex_scheduled: AtomicBool::new(false),
        });

        {
            let mut reg = self.registry.write().await;
            reg.insert(pid, Arc::clone(&project));
        }

        // Lazily start watching this project now that it's loaded.
        if let Some(manager) = self.watch.get() {
            let _ = manager.watch(root, pid).await;
        }

        Ok(project)
    }
}
