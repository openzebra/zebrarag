use std::sync::Mutex;

pub trait ProgressReporter: Send + Sync {
    fn start(&self, total: u64);
    fn inc(&self, n: u64);
    fn finish_with_message(&self, msg: &str);
}

pub struct IndicatifReporter {
    bar: Mutex<Option<indicatif::ProgressBar>>,
}

impl IndicatifReporter {
    pub fn new() -> Self {
        Self {
            bar: Mutex::new(None),
        }
    }
}

impl Default for IndicatifReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for IndicatifReporter {
    fn start(&self, total: u64) {
        let mut guard = self.bar.lock().unwrap();
        if let Some(old) = guard.take() {
            old.finish_and_clear();
        }
        let bar = indicatif::ProgressBar::new(total);
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar()),
        );
        *guard = Some(bar);
    }

    fn inc(&self, n: u64) {
        let guard = self.bar.lock().unwrap();
        if let Some(bar) = guard.as_ref() {
            bar.inc(n);
        }
    }

    fn finish_with_message(&self, msg: &str) {
        let mut guard = self.bar.lock().unwrap();
        if let Some(bar) = guard.take() {
            bar.finish_with_message(msg.to_string());
        }
    }
}

pub struct SilentReporter;

impl ProgressReporter for SilentReporter {
    fn start(&self, _total: u64) {}
    fn inc(&self, _n: u64) {}
    fn finish_with_message(&self, _msg: &str) {}
}
