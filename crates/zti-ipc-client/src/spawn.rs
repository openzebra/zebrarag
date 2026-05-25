use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use tokio::net::UnixStream;
use tokio::time;

use zti_common::paths;

pub async fn connect_or_spawn(
    timeout: Duration,
    model: Option<&str>,
    variant: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
) -> Result<UnixStream> {
    let socket_path = paths::daemon_socket()?;

    if let Ok(stream) = UnixStream::connect(&socket_path).await {
        tracing::debug!("connected to existing daemon");
        return Ok(stream);
    }

    match model {
        Some(m) => tracing::info!("daemon not running, spawning with model {m}..."),
        None => tracing::info!("daemon not running, spawning with daemon default model..."),
    }
    spawn_daemon(model, variant, query_prefix, passage_prefix)?;
    wait_for_socket(&socket_path, timeout).await
}

fn spawn_daemon(
    model: Option<&str>,
    variant: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
) -> Result<()> {
    let exe = std::env::current_exe()?;
    let dir = exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("no parent dir for current exe"))?;

    let exe_stem = exe
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let (daemon_bin, needs_subcommand) = if exe_stem == "zebraindex" {
        (exe, true)
    } else {
        let zebraindex = dir.join("zebraindex");
        if zebraindex.exists() {
            (zebraindex, true)
        } else {
            let fallback = dir.join("zti-daemon");
            if fallback.exists() {
                (fallback, false)
            } else {
                anyhow::bail!(
                    "neither zebraindex nor zti-daemon found in {} — install one",
                    dir.display()
                );
            }
        }
    };

    let log_path = paths::daemon_log()?;
    let log_file = std::fs::File::create(&log_path)?;

    let mut cmd = std::process::Command::new(&daemon_bin);
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(log_file);

    if needs_subcommand {
        cmd.arg("daemon");
    }
    if let Some(m) = model {
        cmd.args(["--model", m]);
    }
    if let Some(v) = variant {
        cmd.args(["--variant", v]);
    }
    if let Some(p) = query_prefix {
        cmd.args(["--query-prefix", p]);
    }
    if let Some(p) = passage_prefix {
        cmd.args(["--passage-prefix", p]);
    }
    cmd.spawn()?;

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
