use anyhow::Result;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub async fn read_frame<R: AsyncRead + Unpin, T: serde::de::DeserializeOwned>(
    reader: &mut R,
) -> Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await?;

    let msg: T = bincode::serde::decode_from_slice(&payload, bincode::config::standard())?.0;
    Ok(msg)
}

pub async fn write_frame<W: AsyncWrite + Unpin, T: serde::Serialize>(
    writer: &mut W,
    msg: &T,
) -> Result<()> {
    let payload = bincode::serde::encode_to_vec(msg, bincode::config::standard())?;
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::*;
    use crate::response::*;

    async fn roundtrip<T>(msg: T) -> T
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let (mut a, mut b) = tokio::io::duplex(64 * 1024);
        write_frame(&mut a, &msg).await.expect("write_frame");
        read_frame(&mut b).await.expect("read_frame")
    }

    #[tokio::test]
    async fn handshake_request_roundtrip() {
        let req = Request::Handshake(HandshakeReq {
            client_version: "0.1.0".to_string(),
            protocol_version: 1,
        });
        let got: Request = roundtrip(req.clone()).await;
        match (req, got) {
            (Request::Handshake(a), Request::Handshake(b)) => {
                assert_eq!(a.client_version, b.client_version);
                assert_eq!(a.protocol_version, b.protocol_version);
            }
            _ => panic!("variant mismatch"),
        }
    }

    #[tokio::test]
    async fn search_request_roundtrip() {
        let req = Request::Search(SearchReq {
            project_root: "/tmp/foo".to_string(),
            query: "rng".to_string(),
            limit: 7,
            offset: Some(3),
            languages: Some(vec!["rust".to_string(), "typescript".to_string()]),
            path_glob: Some("src/**".to_string()),
            refresh_index: false,
            exhaustive: false,
            mode: SearchMode::default(),
        });
        let got: Request = roundtrip(req.clone()).await;
        match (req, got) {
            (Request::Search(a), Request::Search(b)) => {
                assert_eq!(a.project_root, b.project_root);
                assert_eq!(a.query, b.query);
                assert_eq!(a.limit, b.limit);
                assert_eq!(a.offset, b.offset);
                assert_eq!(a.languages, b.languages);
                assert_eq!(a.path_glob, b.path_glob);
                assert_eq!(a.refresh_index, b.refresh_index);
                assert_eq!(a.exhaustive, b.exhaustive);
            }
            _ => panic!("variant mismatch"),
        }
    }

    #[tokio::test]
    async fn index_progress_response_roundtrip() {
        let resp = Response::IndexProgress(IndexingProgress {
            phase: crate::response::IndexPhase::Embed,
            current: 42,
            total: 200,
            message: "batch 3".to_string(),
        });
        let got: Response = roundtrip(resp.clone()).await;
        match (resp, got) {
            (Response::IndexProgress(a), Response::IndexProgress(b)) => {
                assert_eq!(a.phase, b.phase);
                assert_eq!(a.current, b.current);
                assert_eq!(a.total, b.total);
                assert_eq!(a.message, b.message);
            }
            _ => panic!("variant mismatch"),
        }
    }

    #[tokio::test]
    async fn doctor_response_with_checks_roundtrip() {
        let resp = Response::Doctor(Ok(DoctorReport {
            device: "metal".to_string(),
            checks: vec![
                DoctorCheck {
                    name: "model_load".to_string(),
                    status: CheckStatus::Ok,
                    message: "dim=384".to_string(),
                },
                DoctorCheck {
                    name: "disk_free_gib".to_string(),
                    status: CheckStatus::Warn,
                    message: "0.42 GiB free".to_string(),
                },
            ],
        }));
        let got: Response = roundtrip(resp.clone()).await;
        match got {
            Response::Doctor(Ok(report)) => {
                assert_eq!(report.device, "metal");
                assert_eq!(report.checks.len(), 2);
                assert_eq!(report.checks[1].status, CheckStatus::Warn);
            }
            _ => panic!("variant mismatch"),
        }
    }
}
