use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TuiConfig {
    pub default_model: String,
    #[serde(default)]
    pub default_search_method: Option<String>,
    #[serde(default)]
    pub default_dtype: Option<String>,
}

pub fn config_path() -> Result<PathBuf> {
    Ok(zti_common::paths::data_dir()?.join("config.toml"))
}

pub fn load() -> Result<Option<TuiConfig>> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    Ok(Some(toml::from_str(&text)?))
}

pub fn save(model: &str, search_method: Option<&str>, dtype: Option<&str>) -> Result<()> {
    let cfg = TuiConfig {
        default_model: model.to_string(),
        default_search_method: search_method.map(str::to_string),
        default_dtype: dtype.map(str::to_string),
    };
    let path = config_path()?;
    std::fs::write(&path, toml::to_string(&cfg)?)?;
    Ok(())
}
