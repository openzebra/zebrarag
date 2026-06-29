use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;

use anyhow::{Result, bail};
use reqwest::{Client, Response, StatusCode, header};

use crate::provider::RemoteProvider;

const TIMEOUT_SECS: u64 = 30;
const MAX_RETRIES: usize = 4;
const BACKOFF_MS: [u64; MAX_RETRIES] = [500, 1_000, 2_000, 4_000];

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
        let extra = provider.extra_headers();
        let mut headers = header::HeaderMap::with_capacity(extra.len());
        for (name, value) in extra {
            headers.insert(
                header::HeaderName::from_static(name),
                header::HeaderValue::from_static(value),
            );
        }
        let inner = Client::builder()
            .https_only(true)
            .timeout(Duration::from_secs(TIMEOUT_SECS))
            .pool_idle_timeout(Duration::from_secs(90))
            .http2_keep_alive_interval(Duration::from_secs(15))
            .http2_keep_alive_timeout(Duration::from_secs(10))
            .http2_keep_alive_while_idle(true)
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
        tracing::info!(
            provider = %self.provider.label(),
            model,
            items = texts.len(),
            "remote embed: sending batch request"
        );
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
            let url = format!("{}/embeddings", self.provider.base_url());
            tracing::info!(
                provider = %self.provider.label(),
                %url,
                attempt,
                "remote embed: POST request sent"
            );
            let send_result = self
                .inner
                .post(&url)
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
            tracing::info!(
                provider = %self.provider.label(),
                %status,
                attempt,
                "remote embed: response received"
            );
            if status.is_success() {
                // Read the body as text first, then parse manually — so we can
                // log the actual serde/JSON error AND a body excerpt when
                // deserialization fails. `resp.json()` gives only a generic
                // "error decoding response body" with no field/path detail.
                let body_text = resp.text().await?;

                // OpenRouter wraps API errors in HTTP 200 with an
                // {"error":{...}} body (e.g. "Input length exceeds maximum
                // allowed token size"). Detect this before attempting to
                // parse as an embeddings response.
                #[derive(serde::Deserialize)]
                struct ErrorBody {
                    #[serde(default)]
                    message: Option<String>,
                }
                #[derive(serde::Deserialize)]
                struct ErrorResp {
                    #[serde(default)]
                    error: Option<ErrorBody>,
                }
                if let Ok(err_resp) = serde_json::from_str::<ErrorResp>(&body_text)
                    && let Some(err) = err_resp.error
                {
                    let msg = err.message.unwrap_or_else(|| "unknown error".into());
                    bail!("{} API error: {msg}", self.provider.label());
                }

                let mut body: Resp = match serde_json::from_str(&body_text) {
                    Ok(v) => v,
                    Err(e) => {
                        let excerpt: String = body_text.chars().take(500).collect();
                        bail!(
                            "{} embed response decode failed: {e} (body len {}, first 500 chars: {excerpt})",
                            self.provider.label(),
                            body_text.len(),
                        );
                    }
                };
                body.data.sort_unstable_by_key(|d| d.index);
                if body.data.iter().enumerate().any(|(i, d)| d.index != i) {
                    bail!(
                        "{} embed response indices not contiguous: got {} rows",
                        self.provider.label(),
                        body.data.len(),
                    );
                }
                tracing::debug!(
                    provider = %self.provider.label(),
                    rows = body.data.len(),
                    "remote embed: batch completed"
                );
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

    /// List embedding models for this provider. The provider's `models_query`
    /// restricts the result server-side, so callers receive only embedding
    /// models — no client-side filtering needed.
    pub async fn get_models<T>(&self) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let query = self.provider.models_query();
        let url = if query.is_empty() {
            format!("{}/models", self.provider.base_url())
        } else {
            format!("{}/models?{query}", self.provider.base_url())
        };
        let resp = self
            .inner
            .get(url)
            .bearer_auth(self.api_key.as_ref())
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    /// Validate the API key cheaply via `GET /key` (free; no credits required).
    /// Surfaces a bad key on the entry screen instead of as a later daemon crash.
    pub async fn validate_key(&self) -> Result<()> {
        let resp = self
            .inner
            .get(format!(
                "{}{}",
                self.provider.base_url(),
                self.provider.validate_path()
            ))
            .bearer_auth(self.api_key.as_ref())
            .send()
            .await?;
        if resp.status() == StatusCode::UNAUTHORIZED {
            bail!("invalid or unauthorized {} API key", self.provider.label());
        }
        resp.error_for_status()?;
        Ok(())
    }
}
