use std::borrow::Cow;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::oneshot;
use zti_remote_embed::RemoteEmbedEngine;

use crate::{EmbedEngine, Pooled, apply_prefix};

const REMOTE_OUTER_BATCH_MULTIPLIER: usize = 8;

// This enum is stored behind Arc in daemon state; avoiding Box keeps one heap
// allocation and pointer chase out of every embedding call.
#[allow(clippy::large_enum_variant)]
pub enum AnyEmbedEngine {
    Local(EmbedEngine),
    Remote(RemoteEmbedEngine),
}

impl AnyEmbedEngine {
    #[inline]
    pub fn dim(&self) -> usize {
        match self {
            Self::Local(e) => e.dim(),
            Self::Remote(e) => e.dim(),
        }
    }

    #[inline]
    pub fn max_length(&self) -> usize {
        match self {
            Self::Local(e) => e.profile().max_length,
            Self::Remote(e) => e.max_length(),
        }
    }

    #[inline]
    pub const fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }

    /// Submit raw texts to the worker for tokenization + forward pass.
    /// Returns immediately — no synchronous CPU work on the tokio reactor.
    pub fn submit_texts_pooled(&self, texts: &[&str]) -> Result<oneshot::Receiver<Result<Pooled>>> {
        match self {
            Self::Local(e) => {
                if let Some(prefix) = e.profile().passage_prefix.as_deref() {
                    let owned: Arc<[String]> =
                        texts.iter().map(|t| format!("{prefix}{t}")).collect();
                    e.submit_texts(owned)
                } else {
                    let owned: Arc<[String]> = texts.iter().map(|s| (*s).to_string()).collect();
                    e.submit_texts(owned)
                }
            }
            Self::Remote(_) => anyhow::bail!("pipelined submit not available for remote engines"),
        }
    }

    pub fn model_id_str(&self) -> &str {
        match self {
            Self::Local(e) => e.profile().model_id.as_str(),
            Self::Remote(e) => e.model_id(),
        }
    }

    pub fn persisted_model_id(&self) -> Cow<'_, str> {
        match self {
            Self::Local(e) => Cow::Borrowed(e.profile().model_id.as_str()),
            Self::Remote(e) => {
                Cow::Owned(format!("{}{}", e.provider().model_prefix(), e.model_id()))
            }
        }
    }

    pub fn hardware(&self) -> Option<&zti_hw::Hardware> {
        match self {
            Self::Local(e) => Some(e.hardware()),
            Self::Remote(_) => None,
        }
    }

    pub fn device_with_hardware(&self, hardware: &zti_hw::Hardware) -> Result<candle_core::Device> {
        match self {
            Self::Local(e) => e.device(),
            Self::Remote(_) => Ok(zti_hw::candle_device(hardware)),
        }
    }

    pub fn recommended_batch_size(&self) -> usize {
        match self {
            Self::Local(e) => e.recommended_batch_size(),
            Self::Remote(e) => e
                .provider()
                .max_batch_items()
                .saturating_mul(REMOTE_OUTER_BATCH_MULTIPLIER)
                .max(1),
        }
    }

    /// Unified passage embedding that returns `Pooled` directly, bypassing
    /// the `Vec<Vec<f32>>` roundtrip. Prefix application produces `String`s
    /// directly for the worker channel — no intermediate Cow layer.
    pub async fn embed_texts_pooled_async(&self, texts: &[&str]) -> Result<Pooled> {
        let dim = self.dim();
        match self {
            Self::Local(e) => {
                if let Some(prefix) = e.profile().passage_prefix.as_deref() {
                    let owned: Arc<[String]> =
                        texts.iter().map(|t| format!("{prefix}{t}")).collect();
                    e.submit_texts(owned)?
                        .await
                        .map_err(|_| anyhow::anyhow!("embed worker dropped without replying"))?
                } else {
                    e.embed_batch_pooled_async(texts).await
                }
            }
            Self::Remote(e) => {
                let rows = e.embed_texts(texts).await?;
                let batch = rows.len();
                let mut data = Vec::with_capacity(batch.saturating_mul(dim));
                for row in rows {
                    data.extend(row);
                }
                Ok(Pooled { data, dim, batch })
            }
        }
    }

    /// Unified passage embedding: local applies the model's passage prefix, remote sends raw text.
    pub async fn embed_texts_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Local(e) => {
                let prefixed: Vec<Cow<'_, str>> = texts
                    .iter()
                    .map(|text| apply_prefix(text, e.profile().passage_prefix.as_deref()))
                    .collect();
                let refs: Vec<&str> = prefixed.iter().map(|text| text.as_ref()).collect();
                e.embed_batch_async(&refs).await
            }
            Self::Remote(e) => e.embed_texts(texts).await,
        }
    }

    pub async fn embed_query_async(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Local(e) => e.embed_query_async(text).await,
            Self::Remote(e) => e.embed_query(text).await,
        }
    }

    pub async fn embed_passage_async(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Local(e) => e.embed_passage_async(text).await,
            Self::Remote(e) => e.embed_passage(text).await,
        }
    }
}
