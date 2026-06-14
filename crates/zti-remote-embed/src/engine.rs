use std::sync::Arc;

use anyhow::{Result, bail};
use futures::StreamExt;

use crate::client::RemoteEmbedClient;
use crate::models::RemoteModelInfo;
use crate::provider::RemoteProvider;

/// Approximate token ceiling per HTTP request to a remote provider.
/// Tuned to stay well under typical provider limits.
const DEFAULT_BATCH_TOKENS: usize = 100_000;
/// Bytes-per-token estimate for batch sizing math.
const BYTES_PER_TOKEN: usize = 4;
/// Maximum characters per single HTTP request (byte-proxy for the token budget).
const BATCH_CHAR_LIMIT: usize = DEFAULT_BATCH_TOKENS * BYTES_PER_TOKEN;
const DEFAULT_MAX_LENGTH: usize = 4096;
const REMOTE_EMBED_PIPELINE: usize = 4;

async fn probe_dim(client: &RemoteEmbedClient, model_id: &str) -> Result<usize> {
    let rows = client.embed_batch(model_id, &["a"]).await?;
    match rows.into_iter().next() {
        Some(v) if !v.is_empty() => Ok(v.len()),
        _ => bail!("remote probe returned an empty embedding vector"),
    }
}

pub struct RemoteEmbedEngine {
    client: RemoteEmbedClient,
    model_id: Arc<str>,
    dim: usize,
    /// Effective token ceiling for chunking decisions (from model metadata or default).
    max_length: usize,
}

impl RemoteEmbedEngine {
    /// Construct and optionally skip dim-probe when `cached_dim` is known.
    pub async fn connect(
        provider: RemoteProvider,
        api_key: Arc<str>,
        model: &RemoteModelInfo,
        cached_dim: Option<usize>,
    ) -> Result<Self> {
        let client = RemoteEmbedClient::new(provider, api_key)?;
        let dim = match cached_dim {
            Some(d) if d > 0 => d,
            _ => probe_dim(&client, &model.id).await?,
        };
        let max_length = usize::try_from(model.context_length)
            .ok()
            .filter(|len| *len > 0)
            .unwrap_or(DEFAULT_MAX_LENGTH);
        Ok(Self {
            client,
            model_id: Arc::from(model.id.as_str()),
            dim,
            max_length,
        })
    }

    #[inline]
    pub const fn provider(&self) -> RemoteProvider {
        self.client.provider()
    }

    #[inline]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub const fn max_length(&self) -> usize {
        self.max_length
    }

    #[inline]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Dynamic batch sizing: split `texts` into provider-sized sub-batches and
    /// pipeline several HTTP requests while preserving response order. Each batch
    /// is a contiguous borrowed sub-slice of `texts` — no chunk text is copied.
    pub async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let max_items = self.client.provider().max_batch_items().max(1);
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(texts.len().div_ceil(max_items));
        let (mut start, mut batch_chars) = (0usize, 0usize);
        for (i, text) in texts.iter().enumerate() {
            let len = text.len();
            if i > start
                && (i - start >= max_items
                    || batch_chars.saturating_add(len) > BATCH_CHAR_LIMIT)
            {
                ranges.push((start, i));
                start = i;
                batch_chars = 0;
            }
            batch_chars = batch_chars.saturating_add(len);
        }
        ranges.push((start, texts.len()));

        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut stream = futures::stream::iter(ranges)
            .map(|(s, e)| async move {
                match texts.get(s..e) {
                    Some(batch) => self.client.embed_batch(&self.model_id, batch).await,
                    None => Err(anyhow::anyhow!("invalid embed batch range {s}..{e}")),
                }
            })
            .buffered(REMOTE_EMBED_PIPELINE);

        while let Some(rows) = stream.next().await {
            out.extend(rows?);
        }
        Ok(out)
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let rows = self.client.embed_batch(&self.model_id, &[text]).await?;
        rows.into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("remote embed returned no vector"))
    }

    pub async fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_query(text).await
    }
}
