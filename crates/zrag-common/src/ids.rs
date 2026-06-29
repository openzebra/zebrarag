use std::path::Path;

pub fn project_id(root: &Path) -> [u8; 32] {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    blake3::hash(canonical.to_string_lossy().as_bytes()).into()
}
