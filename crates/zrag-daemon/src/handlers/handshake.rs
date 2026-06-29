use zrag_protocol::request::HandshakeReq;
use zrag_protocol::response::{HandshakeResp, Response};

pub fn handle(_req: &HandshakeReq) -> Response {
    Response::Handshake(HandshakeResp {
        ok: true,
        daemon_version: env!("CARGO_PKG_VERSION").to_string(),
        protocol_version: zrag_protocol::PROTOCOL_VERSION,
    })
}
