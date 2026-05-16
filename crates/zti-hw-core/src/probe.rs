use sysinfo::System;

use crate::device::{Device, Hardware};

pub fn probe() -> Hardware {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.refresh_memory();

    let cpus = System::physical_core_count().unwrap_or(1) as u32;
    let mem_total = sys.total_memory();
    let mem_avail = sys.available_memory();

    let device = detect_device();

    Hardware {
        device,
        cpus,
        mem_total,
        mem_avail,
    }
}

fn detect_device() -> Device {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Device::Metal
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Device::Cpu
    }
}
