pub mod detect;
pub mod registry;

pub use detect::detect_from_path;
pub use registry::{Language, frontend_for};
