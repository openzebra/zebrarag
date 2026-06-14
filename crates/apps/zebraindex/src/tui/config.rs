use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TuiConfig {
    pub default_model: String,
    #[serde(default)]
    pub default_search_method: Option<String>,
    #[serde(default)]
    pub default_dtype: Option<String>,
    #[serde(default)]
    pub remote_provider: Option<String>,
    #[serde(default)]
    pub remote_api_key: Option<String>,
    #[serde(default)]
    pub remote_dim_hint: Option<usize>,
}

/// Where a remote provider's API key is persisted at rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteKeyLocation {
    /// No remote key was supplied (local model).
    None,
    /// Stored in the OS keyring — never written to disk in cleartext.
    Keyring,
    /// No keyring backend available; stored as plaintext in the config file.
    Config,
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

fn write(cfg: &TuiConfig) -> Result<()> {
    std::fs::write(config_path()?, toml::to_string(cfg)?)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct SaveConfig<'a> {
    pub model: &'a str,
    pub search_method: Option<&'a str>,
    pub dtype: Option<&'a str>,
    pub remote_provider: Option<&'a str>,
    pub remote_dim_hint: Option<usize>,
}

/// Persist the full config. The remote API key — when present — is routed to the
/// OS keyring and only falls back to plaintext config when no keyring backend is
/// available. This is the single writer of the secret, so no other call site can
/// re-leak it. Returns where the key landed for UI messaging.
pub fn save(args: SaveConfig<'_>, remote_key: Option<&str>) -> Result<RemoteKeyLocation> {
    let location = match (remote_key, args.remote_provider) {
        (Some(key), Some(provider)) if zti_common::secrets::store(provider, key) => {
            RemoteKeyLocation::Keyring
        }
        (Some(_), Some(_)) => RemoteKeyLocation::Config,
        _ => RemoteKeyLocation::None,
    };

    let cfg = TuiConfig {
        default_model: args.model.to_owned(),
        default_search_method: args.search_method.map(str::to_owned),
        default_dtype: args.dtype.map(str::to_owned),
        remote_provider: args.remote_provider.map(str::to_owned),
        remote_api_key: match location {
            RemoteKeyLocation::Config => remote_key.map(str::to_owned),
            RemoteKeyLocation::Keyring | RemoteKeyLocation::None => None,
        },
        remote_dim_hint: args.remote_dim_hint,
    };
    write(&cfg)?;
    Ok(location)
}

/// Load, mutate one field, and write back — without re-touching the API key, so
/// partial updates can never overwrite the keyring decision made at setup time.
fn update(mutate: impl FnOnce(&mut TuiConfig)) -> Result<()> {
    let mut cfg = load()?.unwrap_or_default();
    mutate(&mut cfg);
    write(&cfg)
}

/// Cache the probed embedding dimension for a remote model.
pub fn update_dim_hint(dim: usize) -> Result<()> {
    update(|cfg| cfg.remote_dim_hint = Some(dim))
}

/// Persist a newly chosen search method.
pub fn update_search_method(method: &str) -> Result<()> {
    update(|cfg| cfg.default_search_method = Some(method.to_owned()))
}
