pub const PROTOCOL_VERSION: u32 = 2;

// Wire protocol. Every connection MUST send `Request::Handshake` as its first
// frame and consume the matching `Response::Handshake` before issuing any
// other request — `Stop` included.
pub mod codec;
pub mod render;
pub mod request;
pub mod response;

pub use render::format_search_results;
pub use render::format_search_results_budgeted;
pub use request::Request;
pub use response::Response;
