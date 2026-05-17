use ort::ep::ExecutionProviderDispatch;

pub use zti_hw_core::{BackendKind, Capability, Device, Hardware};

pub fn supported_devices() -> Vec<Device> {
    let mut devs = Vec::with_capacity(4);
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    devs.push(Device::Metal);
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    devs.push(Device::Cuda);
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    devs.push(Device::Vulkan);
    devs.push(Device::Cpu);
    devs
}

pub fn probe() -> Hardware {
    zti_hw_core::probe(&supported_devices())
}

pub fn register() -> Vec<ExecutionProviderDispatch> {
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    eps.extend(zti_hw_metal::register());

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    eps.extend(zti_hw_cuda::register());

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    eps.extend(zti_hw_vulkan::register());

    eps.extend(zti_hw_cpu::register());

    eps
}
