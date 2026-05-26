use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelsRegistry {
    #[serde(rename = "models")]
    pub entries: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub model_id: String,
    pub parameters: String,
    pub technologies: Vec<String>,
    pub description: String,
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
        is_model_downloaded(&self.model_id)
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

pub fn sort_by_hardware(entries: &mut Vec<ModelEntry>) {
    let mut paired: Vec<(ModelEntry, bool)> = Vec::with_capacity(entries.len());
    for e in entries.drain(..) {
        let dl = is_model_downloaded(&e.model_id);
        paired.push((e, dl));
    }
    paired.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (e, _) in paired {
        entries.push(e);
    }
}
