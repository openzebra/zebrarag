use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use tokio::net::UnixStream;
use tokio::time;

use zti_common::paths;

pub async fn connect_or_spawn(timeout: Duration, model: &str) -> Result<UnixStream> {
    let socket_path = paths::daemon_socket()?;

    if let Ok(stream) = UnixStream::connect(&socket_path).await {
        tracing::debug!("connected to existing daemon");
        return Ok(stream);
    }

    tracing::info!("daemon not running, spawning with model {model}...");
    spawn_daemon(model)?;
    wait_for_socket(&socket_path, timeout).await
}

fn spawn_daemon(model: &str) -> Result<()> {
    let exe = std::env::current_exe()?;
    let dir = exe.parent().ok_or_else(|| anyhow::anyhow!("no parent dir for current exe"))?;
    let daemon_path = dir.join("zti-daemon");

    let log_path = paths::daemon_log()?;
    let log_file = std::fs::File::create(&log_path)?;

    std::process::Command::new(&daemon_path)
        .args(["--model", model])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(log_file)
        .spawn()?;

    Ok(())
}

async fn wait_for_socket(socket_path: &PathBuf, timeout: Duration) -> Result<UnixStream> {
    let start = std::time::Instant::now();
    loop {
        match UnixStream::connect(socket_path).await {
            Ok(stream) => return Ok(stream),
            Err(_) => {
                if start.elapsed() > timeout {
                    anyhow::bail!("daemon did not start within {:?}", timeout);
                }
                time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}
