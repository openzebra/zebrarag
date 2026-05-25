use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use serde::Deserialize;
use zti_hw::Device;

#[derive(Debug, Deserialize)]
pub struct ModelsRegistry {
    #[serde(rename = "models")]
    pub entries: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub model_id: String,
    pub optimal_hardware: Vec<String>,
    pub parameters: String,
    pub technologies: Vec<String>,
    pub description: String,
    pub onnx_variants: BTreeMap<String, String>,
}

pub fn is_model_downloaded(model_id: &str) -> bool {
    let dir_name = model_id.replace('/', "_");
    let Ok(models) = zti_common::paths::models_dir() else {
        return false;
    };
    models.join(&dir_name).join(".zti_clone_complete").exists()
}

impl ModelEntry {
    pub fn is_downloaded(&self) -> bool {
        is_model_downloaded(&self.model_id)
    }

    pub fn variant_list(&self) -> Vec<(Arc<str>, Arc<str>)> {
        let mut out = Vec::with_capacity(1 + self.onnx_variants.len());
        out.push((
            Arc::from("Auto (recommended)"),
            Arc::from("System auto-selects best variant for your hardware"),
        ));
        for (name, desc) in &self.onnx_variants {
            out.push((Arc::from(name.as_str()), Arc::from(desc.as_str())));
        }
        out
    }
}

pub fn registry_path() -> Result<PathBuf> {
    Ok(zti_common::paths::data_dir()?.join("models.toml"))
}

pub fn parse(content: &str) -> Result<ModelsRegistry> {
    toml::from_str(content).map_err(Into::into)
}

pub fn load() -> Result<Option<ModelsRegistry>> {
    let path = registry_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)?;
    parse(&content).map(Some)
}

fn device_hardware_strings(device: &Device) -> &'static [&'static str] {
    match device {
        Device::Metal => &["Metal"],
        Device::Cuda => &["CUDA", "CUDA (TensorRT)"],
        Device::Vulkan => &["Vulkan"],
        Device::Cpu => &["CPU", "CPU (AVX512)"],
        Device::Npu => &["NPU", "RockX"],
    }
}

#[allow(clippy::ptr_arg)]
pub fn sort_by_hardware(entries: &mut Vec<ModelEntry>, device: &Device) {
    let hw = device_hardware_strings(device);
    let mut paired: Vec<(ModelEntry, bool, bool)> = Vec::with_capacity(entries.len());
    for e in entries.drain(..) {
        let hw_match = e.optimal_hardware.iter().any(|h| hw.contains(&h.as_str()));
        let dl = is_model_downloaded(&e.model_id);
        paired.push((e, hw_match, dl));
    }
    paired.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    for (e, _, _) in paired {
        entries.push(e);
    }
}
