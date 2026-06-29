pub mod cache;
pub mod method;
pub mod selector;
pub mod usearch_graph;

pub use cache::{AnnCache, AnnHandle};
pub use method::{MethodStats, SearchMethod, SearchParams};
pub use selector::{choose_method, recommend};
pub use usearch_graph::{AnnIndex, AnnIndexBuilder};

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
