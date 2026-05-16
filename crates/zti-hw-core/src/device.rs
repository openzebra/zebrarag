#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
    Npu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

#[derive(Debug, Clone)]
pub struct Capability {
    pub device: Device,
    pub backend: BackendKind,
}

#[derive(Debug, Clone)]
pub struct Hardware {
    pub device: Device,
    pub cpus: u32,
    pub mem_total: u64,
    pub mem_avail: u64,
}
