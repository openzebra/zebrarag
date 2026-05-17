use std::path::PathBuf;

use anyhow::Result;

pub fn data_dir() -> Result<PathBuf> {
    let dir = match std::env::var_os("ZEBRA_DATA_DIR") {
        Some(s) => PathBuf::from(s),
        None => dirs::home_dir()
            .ok_or_else(|| {
                anyhow::anyhow!("ZEBRA_DATA_DIR is unset and home_dir() is unresolvable")
            })?
            .join(".zebra_tree_indexer"),
    };
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn daemon_socket() -> Result<PathBuf> {
    Ok(data_dir()?.join("daemon.sock"))
}

pub fn daemon_pid() -> Result<PathBuf> {
    Ok(data_dir()?.join("daemon.pid"))
}

pub fn daemon_log() -> Result<PathBuf> {
    Ok(data_dir()?.join("daemon.log"))
}

pub fn project_dir(project_id: &[u8; 32]) -> Result<PathBuf> {
    let hex: String = project_id[..8].iter().map(|b| format!("{:02x}", b)).collect();
    let dir = data_dir()?.join("projects").join(hex);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn models_dir() -> Result<PathBuf> {
    let dir = data_dir()?.join("models");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}
