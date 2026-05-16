use ort::ep::ExecutionProviderDispatch;

pub use zti_hw_core::*;

pub fn register() -> Vec<ExecutionProviderDispatch> {
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

    #[cfg(feature = "metal")]
    eps.extend(zti_hw_metal::register());

    #[cfg(feature = "cuda")]
    eps.extend(zti_hw_cuda::register());

    #[cfg(feature = "vulkan")]
    eps.extend(zti_hw_vulkan::register());

    #[cfg(feature = "cpu")]
    eps.extend(zti_hw_cpu::register());

    eps
}
