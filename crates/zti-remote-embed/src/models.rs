use std::sync::Arc;

use anyhow::Result;
use serde::Deserialize;

use crate::client::RemoteEmbedClient;
use crate::provider::RemoteProvider;

#[derive(Debug, Clone, Deserialize)]
pub struct RemoteModelInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub context_length: u32,
}

/// Returns embedding-capable models for a provider.
pub async fn list_models(
    provider: RemoteProvider,
    api_key: &Arc<str>,
) -> Result<Vec<RemoteModelInfo>> {
    match provider {
        RemoteProvider::OpenRouter => list_openrouter_models(api_key).await,
    }
}

/// Case-insensitive substring test without allocating a lowercased copy.
fn contains_ignore_ascii_case(haystack: &str, needle: &[u8]) -> bool {
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

/// Returns OpenRouter embedding-capable models only (id or name contains "embed").
pub async fn list_openrouter_models(api_key: &Arc<str>) -> Result<Vec<RemoteModelInfo>> {
    #[derive(Deserialize)]
    struct Resp {
        data: Vec<RemoteModelInfo>,
    }

    let client = RemoteEmbedClient::new(RemoteProvider::OpenRouter, Arc::clone(api_key))?;
    let resp: Resp = client.get_models().await?;
    let mut models: Vec<RemoteModelInfo> = resp
        .data
        .into_iter()
        .filter(|m| {
            contains_ignore_ascii_case(&m.id, b"embed")
                || contains_ignore_ascii_case(&m.name, b"embed")
        })
        .collect();
    models.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}
