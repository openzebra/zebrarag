use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;

use anyhow::{Result, bail};
use reqwest::{Client, Response, StatusCode, header};

use crate::provider::RemoteProvider;

const TIMEOUT_SECS: u64 = 30;
const MAX_RETRIES: usize = 4;
const BACKOFF_MS: [u64; MAX_RETRIES] = [500, 1_000, 2_000, 4_000];
const OPENROUTER_REFERER: &str = "https://github.com/hicaru/zebra_tree_indexer";
const OPENROUTER_TITLE: &str = "zebraindex";

#[inline]
fn is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn jitter() -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos())
        .unwrap_or(0);
    Duration::from_millis(u64::from(nanos % 250))
}

fn retry_delay(resp: &Response, attempt: usize) -> Duration {
    resp.headers()
        .get(header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_millis(BACKOFF_MS[attempt]))
        .saturating_add(jitter())
}

pub struct RemoteEmbedClient {
    inner: Client,
    provider: RemoteProvider,
    api_key: Arc<str>,
}

impl RemoteEmbedClient {
    #[inline]
    pub const fn provider(&self) -> RemoteProvider {
        self.provider
    }

    pub fn new(provider: RemoteProvider, api_key: Arc<str>) -> Result<Self> {
        let mut headers = header::HeaderMap::with_capacity(2);
        headers.insert(
            header::HeaderName::from_static("http-referer"),
            header::HeaderValue::from_static(OPENROUTER_REFERER),
        );
        headers.insert(
            header::HeaderName::from_static("x-title"),
            header::HeaderValue::from_static(OPENROUTER_TITLE),
        );
        let inner = Client::builder()
            .https_only(true)
            .timeout(Duration::from_secs(TIMEOUT_SECS))
            .default_headers(headers)
            .build()?;
        Ok(Self {
            inner,
            provider,
            api_key,
        })
    }

    /// Batch embed with automatic retry on transient failures (429 / 5xx).
    pub async fn embed_batch(&self, model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        #[derive(serde::Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [&'a str],
        }
        #[derive(serde::Deserialize)]
        struct Resp {
            data: Vec<EmbedData>,
        }
        #[derive(serde::Deserialize)]
        struct EmbedData {
            index: usize,
            embedding: Vec<f32>,
        }

        let mut attempt = 0usize;
        loop {
            let send_result = self
                .inner
                .post(format!("{}/embeddings", self.provider.base_url()))
                .bearer_auth(self.api_key.as_ref())
                .json(&Req {
                    model,
                    input: texts,
                })
                .send()
                .await;
            let resp = match send_result {
                Ok(resp) => resp,
                Err(e) if attempt < MAX_RETRIES => {
                    let delay = Duration::from_millis(BACKOFF_MS[attempt]).saturating_add(jitter());
                    tracing::warn!(
                        attempt,
                        error = %e,
                        ?delay,
                        "remote embed: retrying after transport error"
                    );
                    tokio::time::sleep(delay).await;
                    attempt = attempt.saturating_add(1);
                    continue;
                }
                Err(e) => return Err(e.into()),
            };

            let status = resp.status();
            if status.is_success() {
                let mut body: Resp = resp.json().await?;
                body.data.sort_unstable_by_key(|d| d.index);
                if body.data.iter().enumerate().any(|(i, d)| d.index != i) {
                    bail!(
                        "{} embed response indices not contiguous: got {} rows",
                        self.provider.label(),
                        body.data.len(),
                    );
                }
                return Ok(body.data.into_iter().map(|d| d.embedding).collect());
            }

            if is_retryable(status) && attempt < MAX_RETRIES {
                let delay = retry_delay(&resp, attempt);
                tracing::warn!(
                    attempt,
                    ?status,
                    ?delay,
                    "remote embed: retrying after transient error"
                );
                tokio::time::sleep(delay).await;
                attempt = attempt.saturating_add(1);
                continue;
            }

            let body = resp.text().await.unwrap_or_else(|_| String::new());
            bail!("{} embed error {status}: {body}", self.provider.label());
        }
    }

    /// Model listing — single attempt (failures are surfaced to the TUI immediately).
    pub async fn get_models<T>(&self) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let resp = self
            .inner
            .get(format!("{}/models", self.provider.base_url()))
            .bearer_auth(self.api_key.as_ref())
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }
}
