pub fn format_elapsed(ns: u64) -> String {
    if ns == 0 {
        return String::from("never");
    }
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let diff = now_ns.saturating_sub(ns / 1_000_000_000);
    const MINUTE: u64 = 60;
    const HOUR: u64 = 3600;
    const DAY: u64 = 86400;
    if diff < MINUTE {
        format!("{diff}s ago")
    } else if diff < HOUR {
        format!("{}m ago", diff / MINUTE)
    } else if diff < DAY {
        format!("{}h ago", diff / HOUR)
    } else {
        format!("{}d ago", diff / DAY)
    }
}
