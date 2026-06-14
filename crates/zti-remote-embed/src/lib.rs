pub mod client;
pub mod engine;
pub mod models;
pub mod provider;

pub use engine::RemoteEmbedEngine;
pub use models::{RemoteModelInfo, RemoteModelPricing, fetch_model_info, list_models};
pub use provider::{RemoteProvider, is_embedding_model};
