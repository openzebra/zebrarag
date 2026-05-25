use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TuiConfig {
    pub default_model: String,
    pub default_variant: String,
}

pub fn config_path() -> Result<PathBuf> {
    Ok(zti_common::paths::data_dir()?.join("config.json"))
}

pub fn load() -> Result<Option<TuiConfig>> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)?;
    let cfg: TuiConfig = serde_json::from_slice(&bytes)?;
    Ok(Some(cfg))
}

pub fn save(model: &str, variant: &str) -> Result<()> {
    let cfg = TuiConfig {
        default_model: model.to_string(),
        default_variant: variant.to_string(),
    };
    let path = config_path()?;
    let json = serde_json::to_vec(&cfg)?;
    std::fs::write(&path, &json)?;
    Ok(())
}
