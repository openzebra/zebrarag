use sysinfo::System;

use crate::device::{Device, Hardware};

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
    }
}

fn select_device(supported: &[Device]) -> Device {
    for d in [&Device::Metal, &Device::Cuda] {
        if supported.contains(d) && hardware_supports(d) {
            return *d;
        }
    }
    Device::Cpu
}

fn hardware_supports(d: &Device) -> bool {
    match d {
        Device::Cpu => true,
        Device::Metal => cfg!(target_os = "macos"),
        Device::Cuda => cfg!(any(target_os = "linux", target_os = "windows")),
    }
}
