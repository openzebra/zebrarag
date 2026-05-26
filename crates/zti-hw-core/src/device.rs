use std::borrow::Cow;
use std::sync::atomic::{AtomicU8, Ordering};

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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpStatus {
    Active,
    Fallback,
    CpuOnly,
    #[default]
    Unknown,
}

impl EpStatus {
    const fn to_u8(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Fallback => 1,
            Self::CpuOnly => 2,
            Self::Unknown => 3,
        }
    }

    const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Active,
            1 => Self::Fallback,
            2 => Self::CpuOnly,
            _ => Self::Unknown,
        }
    }

    pub fn device_label(&self, device: &Device) -> Cow<'static, str> {
        let base = device.as_str();
        match self {
            Self::Active => Cow::Owned(format!("{base} (gpu/ane)")),
            Self::Fallback => Cow::Owned(format!("{base} (cpu fallback)")),
            Self::CpuOnly | Self::Unknown => Cow::Borrowed(base),
        }
    }
}

pub struct AtomicEpStatus(AtomicU8);

impl AtomicEpStatus {
    pub const fn new(status: EpStatus) -> Self {
        Self(AtomicU8::new(status.to_u8()))
    }

    pub fn get(&self) -> EpStatus {
        EpStatus::from_u8(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, status: EpStatus) {
        self.0.store(status.to_u8(), Ordering::Relaxed);
    }
}

impl Default for AtomicEpStatus {
    fn default() -> Self {
        Self::new(EpStatus::Unknown)
    }
}

impl Clone for AtomicEpStatus {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

impl std::fmt::Debug for AtomicEpStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get())
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

pub struct Hardware {
    pub device: Device,
    pub cpus: usize,
    pub mem_total: u64,
    pub mem_avail: u64,
    pub ep_status: AtomicEpStatus,
}

impl Clone for Hardware {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            cpus: self.cpus,
            mem_total: self.mem_total,
            mem_avail: self.mem_avail,
            ep_status: self.ep_status.clone(),
        }
    }
}

impl Default for Hardware {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            cpus: 1,
            mem_total: 0,
            mem_avail: 0,
            ep_status: AtomicEpStatus::default(),
        }
    }
}
