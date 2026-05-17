use ort::ep::ExecutionProviderDispatch;

pub use zti_hw_core::{BackendKind, Capability, Device, Hardware};

/// Devices whose backend crates are linked into this build.
pub fn supported_devices() -> Vec<Device> {
    let mut devs = Vec::new();
    #[cfg(feature = "metal")]
    devs.push(Device::Metal);
    #[cfg(feature = "cuda")]
    devs.push(Device::Cuda);
    #[cfg(feature = "vulkan")]
    devs.push(Device::Vulkan);
    #[cfg(feature = "cpu")]
    devs.push(Device::Cpu);
    devs
}

pub fn probe() -> Hardware {
    zti_hw_core::probe(&supported_devices())
}

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
