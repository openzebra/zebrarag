pub mod fusion;
pub mod indexer;
pub mod manifest;
pub mod pdf_chunk;
pub mod progress;
pub mod search;

/// Re-exports of otherwise-internal helpers for integration tests and the
/// `testing` feature. Gated so the symbols only exist under `cfg(test)` or
/// `--features testing`, keeping the release API surface clean.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    pub use crate::manifest::{PdfPageMeta, assemble_pdf_contents};
    pub use crate::pdf_chunk::pack_pdf_pages;
}

#[cfg(test)]
mod alloc_counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static BYTES: AtomicUsize = AtomicUsize::new(0);
    pub static COUNT: AtomicUsize = AtomicUsize::new(0);

    pub struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            BYTES.fetch_add(layout.size(), Ordering::SeqCst);
            COUNT.fetch_add(1, Ordering::SeqCst);
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
            unsafe { System.dealloc(_ptr, _layout) }
        }
    }

    pub fn snapshot() -> (usize, usize) {
        (COUNT.load(Ordering::SeqCst), BYTES.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
#[global_allocator]
static GLOBAL: alloc_counting::Counting = alloc_counting::Counting;
