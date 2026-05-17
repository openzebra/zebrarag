#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
    Npu,
}

impl Device {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
            Device::Cuda => "cuda",
            Device::Vulkan => "vulkan",
            Device::Npu => "npu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

#[derive(Debug, Clone, Copy)]
pub struct Capability {
    pub device: Device,
    pub backend: BackendKind,
}

#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    pub device: Device,
    pub cpus: usize,
    pub mem_total: u64,
    pub mem_avail: u64,
}
