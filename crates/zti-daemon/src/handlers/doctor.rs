use fs2::available_space;

use zti_hw::EpStatus;
use zti_protocol::request::DoctorReq;
use zti_protocol::response::{CheckStatus, DoctorCheck, DoctorReport, Response};

use crate::state::DaemonState;

pub async fn handle(req: &DoctorReq, state: &DaemonState) -> Response {
    let mut checks: Vec<DoctorCheck> = Vec::with_capacity(8);

    let canonical_root = req
        .project_root
        .as_deref()
        .and_then(|r| std::path::Path::new(r).canonicalize().ok());

    let engine = if let Some(canon) = canonical_root.as_deref() {
        let pid = zti_common::ids::project_id(canon);
        let root_str = canon.to_string_lossy();
        let model_id = match state.load_or_open(&root_str).await {
            Ok(project) => match project.db.projects_table().await {
                Ok(table) => match table.get(&pid).await {
                    Ok(Some(row)) if !row.model_id.is_empty() => Some(row.model_id),
                    _ => None,
                },
                Err(_) => None,
            },
            Err(_) => None,
        };
        match model_id {
            Some(mid) => state
                .engine_for_model(&mid)
                .await
                .unwrap_or_else(|_| state.primary_engine()),
            None => state.primary_engine(),
        }
    } else {
        state.primary_engine()
    };

    let onnx_path = &engine.profile().onnx_path;
    if !onnx_path.exists() {
        checks.push(error_check(
            "model_load",
            format!("ONNX file missing at {}", onnx_path.display()),
        ));
    } else {
        match engine.embed_batch_async(&["hello"]).await {
            Ok(embs) => match embs.first() {
                Some(emb) if emb.len() == engine.dim() => checks.push(ok_check(
                    "model_load",
                    format!("dim={} via {}", emb.len(), onnx_path.display()),
                )),
                Some(emb) => checks.push(error_check(
                    "model_load",
                    format!(
                        "dim mismatch: profile={} probe={}",
                        engine.dim(),
                        emb.len()
                    ),
                )),
                None => checks.push(error_check("model_load", "probe returned no embedding")),
            },
            Err(e) => checks.push(error_check(
                "model_load",
                format!("embed probe failed: {}", e),
            )),
        }
    }

    match engine.hardware().ep_status.get() {
        EpStatus::Active => checks.push(ok_check("ep_verify", "CoreML GPU/ANE active")),
        EpStatus::Fallback => checks.push(DoctorCheck {
            name: "ep_verify".to_string(),
            status: CheckStatus::Warn,
            message: "CoreML registered but ops running on CPU. Try --variant fp32.".to_string(),
        }),
        EpStatus::CpuOnly => checks.push(ok_check("ep_verify", "CPU execution provider")),
        EpStatus::Unknown => checks.push(DoctorCheck {
            name: "ep_verify".to_string(),
            status: CheckStatus::Warn,
            message: "EP status not verified".to_string(),
        }),
    }

    // -- data_dir_writable: probe by writing a tiny file under data_dir and
    //    deleting it. Catches read-only filesystems, broken permissions, etc.
    match zti_common::paths::data_dir() {
        Ok(dir) => {
            let probe = dir.join(".doctor-write-probe");
            match std::fs::write(&probe, b"ok") {
                Ok(()) => {
                    let _ = std::fs::remove_file(&probe);
                    checks.push(ok_check("data_dir_writable", dir.display().to_string()));
                }
                Err(e) => checks.push(error_check(
                    "data_dir_writable",
                    format!("{}: {}", dir.display(), e),
                )),
            }

            // -- disk_free: rough warning threshold = 1 GiB.
            match available_space(&dir) {
                Ok(bytes) => {
                    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    let status = if gib < 1.0 {
                        CheckStatus::Warn
                    } else {
                        CheckStatus::Ok
                    };
                    checks.push(DoctorCheck {
                        name: "disk_free_gib".to_string(),
                        status,
                        message: format!("{:.2} GiB free at {}", gib, dir.display()),
                    });
                }
                Err(e) => checks.push(error_check("disk_free_gib", e.to_string())),
            }
        }
        Err(e) => checks.push(error_check("data_dir_writable", e.to_string())),
    }

    // -- per-project DB probes: db_open, chunks_count, files_count.
    if let Some(canon) = canonical_root.as_deref() {
        let pid = zti_common::ids::project_id(canon);
        let root_str = canon.to_string_lossy();
        let db_path = match zti_common::paths::project_dir(&pid) {
            Ok(p) => p.join("lance"),
            Err(e) => {
                checks.push(error_check("db_open", e.to_string()));
                return finalize(&engine, checks);
            }
        };

        match state.load_or_open(&root_str).await {
            Ok(project) => {
                checks.push(ok_check("db_open", db_path.display().to_string()));
                match project.db.chunks_table(engine.dim()).await {
                    Ok(t) => match t.len().await {
                        Ok(n) => checks.push(ok_check("chunks_count", n.to_string())),
                        Err(e) => checks.push(error_check("chunks_count", e.to_string())),
                    },
                    Err(e) => checks.push(error_check("chunks_count", e.to_string())),
                }
                match project.db.files_table().await {
                    Ok(t) => match t.len().await {
                        Ok(n) => checks.push(ok_check("files_count", n.to_string())),
                        Err(e) => checks.push(error_check("files_count", e.to_string())),
                    },
                    Err(e) => checks.push(error_check("files_count", e.to_string())),
                }
            }
            Err(e) => checks.push(error_check(
                "db_open",
                format!("{}: {}", db_path.display(), e),
            )),
        }
    }

    finalize(&engine, checks)
}

fn finalize(engine: &zti_embed::EmbedEngine, checks: Vec<DoctorCheck>) -> Response {
    let ep = engine.hardware().ep_status.get();
    let device = ep.device_label(&engine.hardware().device).into_owned();
    Response::Doctor(Ok(DoctorReport {
        device,
        checks,
    }))
}

fn ok_check(name: &str, message: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name: name.to_string(),
        status: CheckStatus::Ok,
        message: message.into(),
    }
}

fn error_check(name: &str, message: impl Into<String>) -> DoctorCheck {
    DoctorCheck {
        name: name.to_string(),
        status: CheckStatus::Err,
        message: message.into(),
    }
}
