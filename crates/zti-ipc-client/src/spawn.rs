use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use fs2::FileExt;
use tokio::net::UnixStream;
use tokio::time;

use zti_common::paths;

pub async fn kill_daemon() -> Result<()> {
    let pid_path = paths::daemon_pid()?;
    let socket_path = paths::daemon_socket()?;

    if let Ok(pid_str) = std::fs::read_to_string(&pid_path)
        && let Ok(pid) = pid_str.trim().parse::<u32>()
    {
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status();
        time::sleep(Duration::from_millis(200)).await;
    }

    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_file(&pid_path);

    Ok(())
}

pub async fn connect_or_spawn(
    timeout: Duration,
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
    model_dtype: Option<&str>,
    remote_api_key: Option<&str>,
    remote_dim_hint: Option<usize>,
) -> Result<UnixStream> {
    let socket_path = paths::daemon_socket()?;

    if let Ok(stream) = UnixStream::connect(&socket_path).await {
        tracing::debug!("connected to existing daemon");
        return Ok(stream);
    }

    let pid_path = paths::daemon_pid()?;
    let pid_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&pid_path)
        .ok();
    let should_spawn = pid_file
        .as_ref()
        .and_then(|f| f.try_lock_exclusive().ok())
        .is_some();

    if should_spawn {
        match model {
            Some(m) => tracing::info!("daemon not running, spawning with model {m}..."),
            None => tracing::info!("daemon not running, spawning (no model specified)..."),
        }
        spawn_daemon(
            model,
            query_prefix,
            passage_prefix,
            model_dtype,
            remote_api_key,
            remote_dim_hint,
        )?;
        drop(pid_file);
    } else {
        tracing::debug!("daemon already spawning, waiting for socket...");
    }

    wait_for_socket(&socket_path, timeout).await
}

fn spawn_daemon(
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
    model_dtype: Option<&str>,
    remote_api_key: Option<&str>,
    remote_dim_hint: Option<usize>,
) -> Result<()> {
    let exe = std::env::current_exe()?;
    let dir = exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("no parent dir for current exe"))?;

    let exe_stem = exe.file_stem().and_then(|s| s.to_str()).unwrap_or("");

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
        let m = model.ok_or_else(|| anyhow::anyhow!("--model is required to spawn the daemon"))?;
        cmd.args(["--model", m]);
    } else if let Some(m) = model {
        cmd.args(["--model", m]);
    }
    if let Some(p) = query_prefix {
        cmd.args(["--query-prefix", p]);
    }
    if let Some(p) = passage_prefix {
        cmd.args(["--passage-prefix", p]);
    }
    if let Some(d) = model_dtype {
        cmd.args(["--model-dtype", d]);
    }
    if let Some(key) = remote_api_key
        && let Some((provider, _)) = model.and_then(zti_remote_embed::RemoteProvider::from_model_id)
    {
        cmd.env(provider.env_var(), key);
    }
    if let Some(dim) = remote_dim_hint {
        cmd.env("ZEBRA_REMOTE_DIM_HINT", dim.to_string());
    }
    cmd.spawn()?;

    Ok(())
}

async fn wait_for_socket(socket_path: &Path, timeout: Duration) -> Result<UnixStream> {
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
