use std::sync::Arc;

use anyhow::Result;
use serde::Deserialize;

use crate::client::RemoteEmbedClient;
use crate::provider::{RemoteProvider, is_embedding_model};

#[derive(Debug, Clone, Deserialize)]
pub struct RemoteModelPricing {
    #[serde(default)]
    pub prompt: String,
}

impl RemoteModelPricing {
    /// Dollar cost per million tokens. Returns 0.0 when the prompt price
    /// string is missing, empty, or unparseable.
    pub fn price_per_million(&self) -> f64 {
        self.prompt
            .parse::<f64>()
            .unwrap_or(0.0)
            .mul_add(1_000_000.0, 0.0)
    }

    #[inline]
    pub fn is_free(&self) -> bool {
        self.price_per_million() == 0.0
    }

    /// Human-readable price label: `"FREE"` when the model costs nothing,
    /// `"$0.20/M tok"` otherwise (3 decimal places for sub-cent pricing).
    pub fn format_price(&self) -> String {
        let price = self.price_per_million();
        if price == 0.0 {
            "FREE".to_string()
        } else if price >= 0.01 {
            format!("${price:.2}/M tok")
        } else {
            format!("${price:.3}/M tok")
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RemoteModelInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub context_length: u32,
    #[serde(default)]
    pub pricing: Option<RemoteModelPricing>,
}

/// Returns embedding-capable models for a provider.
///
/// Providers that filter server-side (via [`RemoteProvider::models_query`])
/// return only embedding models; the rest expose an unfiltered `/models`
/// listing, so those are filtered client-side with [`is_embedding_model`].
/// The key is validated first so a bad key fails here, not at daemon launch.
pub async fn list_models(
    provider: RemoteProvider,
    api_key: &Arc<str>,
) -> Result<Vec<RemoteModelInfo>> {
    #[derive(Deserialize)]
    struct Resp {
        data: Vec<RemoteModelInfo>,
    }

    let client = RemoteEmbedClient::new(provider, Arc::clone(api_key))?;
    client.validate_key().await?;
    let resp: Resp = client.get_models().await?;
    let mut models = resp.data;
    if provider.requires_client_side_embedding_filter() {
        models.retain(|m| is_embedding_model(&m.id));
    }
    models.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

/// Fetch metadata for a single model id (e.g. its real `context_length`).
/// Errors when the listing fails or the id isn't an embedding model for this
/// provider — surfacing a bad model id at launch instead of letting the engine
/// silently run with default settings.
pub async fn fetch_model_info(
    provider: RemoteProvider,
    api_key: &Arc<str>,
    model_id: &str,
) -> Result<RemoteModelInfo> {
    list_models(provider, api_key)
        .await?
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "model '{model_id}' is not an available {} embedding model",
                provider.label()
            )
        })
}
