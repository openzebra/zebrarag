use sysinfo::System;

use crate::device::{AtomicEpStatus, Device, EpStatus, Hardware};

/// Probe physical hardware. `supported` lists devices whose backend crates are
/// linked into this build (set by the umbrella `zti-hw` crate from feature
/// flags); `probe` picks the highest-priority entry that is both supported and
/// physically present, falling back to CPU.
pub fn probe(supported: &[Device]) -> Hardware {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.refresh_memory();

    let cpus = System::physical_core_count().unwrap_or(1);
    let mem_total = sys.total_memory();
    let mem_avail = sys.available_memory();

    let device = select_device(supported);

    Hardware {
        device,
        cpus,
        mem_total,
        mem_avail,
        ep_status: AtomicEpStatus::new(EpStatus::Unknown),
    }
}

fn select_device(supported: &[Device]) -> Device {
    for d in [Device::Metal, Device::Cuda, Device::Vulkan, Device::Npu] {
        if supported.contains(&d) && hardware_supports(d) {
            return d;
        }
    }
    Device::Cpu
}

fn hardware_supports(d: Device) -> bool {
    match d {
        Device::Cpu => true,
        Device::Metal => cfg!(all(target_os = "macos", target_arch = "aarch64")),
        Device::Cuda => cfg!(any(target_os = "linux", target_os = "windows")),
        Device::Vulkan => true,
        Device::Npu => false,
    }
}
