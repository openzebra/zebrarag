use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{OnceCell, RwLock};

use crate::usearch_graph::AnnIndex;

pub type ProjectId = [u8; 32];
pub type AnnHandle = Arc<AnnIndex>;

#[derive(Default)]
pub struct AnnCache {
    inner: RwLock<HashMap<ProjectId, Arc<OnceCell<AnnHandle>>>>,
}

impl AnnCache {
    #[inline]
    pub async fn peek(&self, pid: &ProjectId) -> Option<AnnHandle> {
        let map = self.inner.read().await;
        map.get(pid).and_then(|cell| cell.get().cloned())
    }

    pub async fn get_or_build<F, Fut, E>(&self, pid: ProjectId, builder: F) -> Result<AnnHandle, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<AnnIndex, E>>,
    {
        let cell = {
            let mut map = self.inner.write().await;
            map.entry(pid)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };
        cell.get_or_try_init(|| async move {
            let g = builder().await?;
            Ok::<AnnHandle, E>(Arc::new(g))
        })
        .await
        .cloned()
    }

    pub async fn invalidate(&self, pid: &ProjectId) {
        self.inner.write().await.remove(pid);
    }
}
