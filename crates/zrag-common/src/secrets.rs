use std::collections::HashMap;
use std::sync::OnceLock;

const SERVICE: &str = "zebrarag";

#[cfg(target_os = "macos")]
fn configure_store() -> keyring_core::Result<()> {
    use apple_native_keyring_store::keychain::Store;
    keyring_core::set_default_store(Store::new_with_configuration(&HashMap::new())?);
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
fn configure_store() -> keyring_core::Result<()> {
    use dbus_secret_service_keyring_store::Store;
    keyring_core::set_default_store(Store::new_with_configuration(&HashMap::new())?);
    Ok(())
}

#[cfg(target_os = "windows")]
fn configure_store() -> keyring_core::Result<()> {
    use windows_native_keyring_store::Store;
    keyring_core::set_default_store(Store::new_with_configuration(&HashMap::new())?);
    Ok(())
}

#[cfg(not(any(
    target_os = "freebsd",
    target_os = "linux",
    target_os = "macos",
    target_os = "windows",
)))]
fn configure_store() -> keyring_core::Result<()> {
    Err(keyring_core::Error::NotSupportedByStore(
        "no native keyring store configured for this platform".to_string(),
    ))
}

fn keyring_ready() -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| configure_store().is_ok())
}

pub fn available() -> bool {
    keyring_ready()
}

/// Store a secret in the OS keyring.
///
/// Returns `false` when no usable backend is available, allowing callers to
/// fall back to explicit local config storage.
pub fn store(account: &str, secret: &str) -> bool {
    keyring_ready()
        && keyring_core::Entry::new(SERVICE, account)
            .and_then(|entry| entry.set_password(secret))
            .is_ok()
}

/// Retrieve a secret from the OS keyring, if present and accessible.
pub fn retrieve(account: &str) -> Option<String> {
    if !keyring_ready() {
        return None;
    }
    keyring_core::Entry::new(SERVICE, account)
        .and_then(|entry| entry.get_password())
        .ok()
}

pub fn delete(account: &str) {
    if !keyring_ready() {
        return;
    }
    if let Ok(entry) = keyring_core::Entry::new(SERVICE, account) {
        let _ = entry.delete_credential();
    }
}
