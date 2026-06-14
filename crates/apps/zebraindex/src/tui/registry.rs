use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

pub use zti_remote_embed::RemoteProvider;

#[derive(Debug, Deserialize)]
pub struct ModelsRegistry {
    #[serde(rename = "models")]
    pub entries: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum ModelSource {
    #[default]
    Local,
    Remote(RemoteProvider),
}

#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub model_id: String,
    pub parameters: String,
    pub technologies: Vec<String>,
    pub description: String,
    #[serde(default, skip_deserializing)]
    pub source: ModelSource,
}

pub fn is_model_downloaded(model_id: &str) -> bool {
    let dir_name = model_id.replace('/', "_");
    let Ok(models) = zti_common::paths::models_dir() else {
        return false;
    };
    models.join(&dir_name).join(".zti_clone_complete").exists()
}

impl ModelEntry {
    pub fn is_downloaded(&self) -> bool {
        match self.source {
            ModelSource::Local => is_model_downloaded(&self.model_id),
            ModelSource::Remote(_) => false,
        }
    }
}

pub fn openrouter_sentinel() -> ModelEntry {
    ModelEntry {
        model_id: String::from("openrouter"),
        parameters: String::from("remote"),
        technologies: vec![String::from("API")],
        description: String::from("Embeddings via OpenRouter API — no local download required"),
        source: ModelSource::Remote(RemoteProvider::OpenRouter),
    }
}

pub fn registry_path() -> Result<PathBuf> {
    Ok(zti_common::paths::data_dir()?.join("models.toml"))
}

pub fn parse(content: &str) -> Result<ModelsRegistry> {
    toml::from_str(content).map_err(Into::into)
}

pub fn load() -> Result<Option<ModelsRegistry>> {
    let path = registry_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)?;
    parse(&content).map(Some)
}

pub fn sort_by_hardware(entries: &mut [ModelEntry]) {
    entries.sort_by_key(|entry| {
        let rank = match entry.source {
            ModelSource::Remote(_) => 2,
            ModelSource::Local if is_model_downloaded(&entry.model_id) => 1,
            ModelSource::Local => 0,
        };
        std::cmp::Reverse(rank)
    });
}
