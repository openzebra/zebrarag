use std::time::Duration;

use anyhow::Result;
use tokio::net::UnixStream;

use zti_protocol::codec::{read_frame, write_frame};
use zti_protocol::request::{HandshakeReq, Request};
use zti_protocol::response::{HandshakeResp, Response};

use crate::spawn::connect_or_spawn;

pub struct Client {
    stream: UnixStream,
}

impl Client {
    pub async fn connect(timeout: Duration) -> Result<Self> {
        let stream = connect_or_spawn(timeout).await?;
        Ok(Self { stream })
    }

    pub async fn handshake(&mut self) -> Result<HandshakeResp> {
        let req = Request::Handshake(HandshakeReq {
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: 1,
        });
        write_frame(&mut self.stream, &req).await?;
        let resp: Response = read_frame(&mut self.stream).await?;
        match resp {
            Response::Handshake(h) => Ok(h),
            other => anyhow::bail!("expected Handshake response, got {:?}", other),
        }
    }

    pub async fn request(&mut self, req: Request) -> Result<Response> {
        write_frame(&mut self.stream, &req).await?;
        let resp: Response = read_frame(&mut self.stream).await?;
        Ok(resp)
    }

    pub async fn request_streaming<F>(&mut self, req: Request, mut on_progress: F) -> Result<Response>
    where
        F: FnMut(Response),
    {
        write_frame(&mut self.stream, &req).await?;
        loop {
            let resp: Response = read_frame(&mut self.stream).await?;
            match &resp {
                Response::Index(_) | Response::Stop(_) => return Ok(resp),
                Response::IndexProgress(_) => on_progress(resp),
                _ => return Ok(resp),
            }
        }
    }
}
