use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use zti_hw::{Device, Hardware};

use crate::pooling::PoolingStrategy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub model_id: String,
    pub onnx_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dim: usize,
    pub max_length: usize,
    pub pooling: PoolingStrategyEnum,
    pub query_prefix: Option<String>,

    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PoolingStrategyEnum {
    Mean,
    Cls,
}

impl From<PoolingStrategyEnum> for PoolingStrategy {
    fn from(v: PoolingStrategyEnum) -> Self {
        match v {
            PoolingStrategyEnum::Mean => PoolingStrategy::Mean,
            PoolingStrategyEnum::Cls => PoolingStrategy::Cls,
        }
    }
}

pub struct ResolvedModel {
    pub onnx_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
}

impl ModelConfig {
    #[inline]
    pub fn ffn_size(&self) -> usize {
        self.intermediate_size.unwrap_or(self.hidden_size * 4)
    }
}

impl<'de> serde::Deserialize<'de> for ModelConfig {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(de)?;
        fn u64_or<E: serde::de::Error>(v: &serde_json::Value, primary: &str, alt: &str) -> std::result::Result<u64, E> {
            v.get(primary)
                .or_else(|| v.get(alt))
                .and_then(|v| v.as_u64())
                .ok_or_else(|| E::custom(format!("missing field `{}` or `{}`", primary, alt)))
        }
        Ok(ModelConfig {
            hidden_size: u64_or(&v, "hidden_size", "n_embd")? as usize,
            num_hidden_layers: u64_or(&v, "num_hidden_layers", "n_layer")? as usize,
            intermediate_size: v
                .get("intermediate_size")
                .or_else(|| v.get("n_inner"))
                .or_else(|| v.get("inner_dim"))
                .or_else(|| v.get("dim_feedforward"))
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            max_position_embeddings: u64_or(&v, "max_position_embeddings", "n_positions")? as usize,
            num_attention_heads: u64_or(&v, "num_attention_heads", "n_head")? as usize,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerCfg {
    #[serde(default)]
    pub model_max_length: Option<serde_json::Value>,
}

impl TokenizerCfg {
    pub fn effective_max_length(&self) -> Option<usize> {
        let v = self.model_max_length.as_ref()?;
        if let Some(u) = v.as_u64() {
            return (u <= i64::MAX as u64).then_some(u as usize);
        }
        if let Some(f) = v.as_f64()
            && f.is_finite()
            && f >= 1.0
            && f <= i64::MAX as f64
        {
            return Some(f as usize);
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum OnnxVariant {
    Auto,
    Fp32,
    Fp16,
    O4,
    Int8,
    Uint8,
    Quantized,
    Q4,
    Q4f16,
    Bnb4,
}

impl OnnxVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto      => "auto",
            Self::Fp32      => "fp32",
            Self::Fp16      => "fp16",
            Self::O4        => "o4",
            Self::Int8      => "int8",
            Self::Uint8     => "uint8",
            Self::Quantized => "quantized",
            Self::Q4        => "q4",
            Self::Q4f16     => "q4f16",
            Self::Bnb4      => "bnb4",
        }
    }
}

impl std::fmt::Display for OnnxVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn classify(lower_filename: &str) -> Option<OnnxVariant> {
    use OnnxVariant::*;

    let stem = lower_filename.strip_suffix(".onnx")?;
    if stem == "model" {
        return Some(Fp32);
    }
    let body = stem
        .strip_prefix("model_")
        .or_else(|| stem.strip_prefix("model-"))
        .unwrap_or(stem);

    let mut best: Option<(OnnxVariant, u8)> = None;
    for tok in body.split(['_', '-', '.']) {
        let cur = match tok {
            "q4f16"               => (Q4f16, 0),
            "bnb4"                => (Bnb4, 1),
            "q4"                  => (Q4, 2),
            "fp16"                => (Fp16, 3),
            "uint8"               => (Uint8, 4),
            "int8" | "qint8"     => (Int8, 5),
            "o4"                  => (O4, 6),
            t if t.starts_with("quant") => (Quantized, 7),
            _ => continue,
        };
        if best.is_none_or(|(_, p)| cur.1 < p) {
            best = Some(cur);
        }
    }
    best.map(|(v, _)| v)
}

const GIB: u64 = 1024 * 1024 * 1024;

fn auto_variant_order(hw: &Hardware) -> &'static [OnnxVariant] {
    use OnnxVariant::*;

    match hw.device {
        Device::Cuda   => &[O4, Fp16, Q4f16, Q4, Int8, Quantized, Fp32],
        Device::Metal  => &[O4, Fp16, Q4f16, Int8, Quantized, Fp32],
        Device::Vulkan => &[O4, Fp16, Q4f16, Int8, Quantized, Fp32],
        Device::Npu    => &[Int8, Uint8, Quantized, Fp16, O4, Fp32],
        Device::Cpu => match hw.mem_total {
            m if m <  4 * GIB => &[Q4, Q4f16, Int8, Uint8, Quantized, Fp16, Fp32],
            m if m <  8 * GIB => &[Int8, Quantized, Q4f16, Fp16, Fp32],
            _                 => &[Int8, Quantized, Fp16, Fp32],
        },
    }
}

const TOKENIZER_CANDIDATES: &[&str] = &["tokenizer.json", "onnx/tokenizer.json"];

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    serde_json::from_reader(reader)
        .with_context(|| format!("parsing {}", path.display()))
}

#[derive(Debug, Clone, Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: Option<bool>,
    #[serde(default)]
    pooling_mode_mean_tokens: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct StConfig {
    #[serde(default)]
    prompts: Option<serde_json::Value>,
}

fn read_pooling_from_metadata(dir: &Path) -> PoolingStrategyEnum {
    let pooling_path = dir.join("1_Pooling").join("config.json");
    if let Ok(cfg) = read_json::<PoolingConfig>(&pooling_path) {
        if cfg.pooling_mode_cls_token == Some(true) {
            return PoolingStrategyEnum::Cls;
        }
        if cfg.pooling_mode_mean_tokens == Some(true) {
            return PoolingStrategyEnum::Mean;
        }
    }
    PoolingStrategyEnum::Mean
}

fn read_query_prefix_from_metadata(dir: &Path) -> Option<String> {
    let st_path = dir.join("config_sentence_transformers.json");
    let cfg = read_json::<StConfig>(&st_path).ok()?;
    let prompts = cfg.prompts?;
    if let Some(obj) = prompts.as_object()
        && let Some(query_prompt) = obj.get("query").and_then(|v| v.as_str())
    {
        return Some(query_prompt.to_string());
    }
    None
}

pub fn resolve_profile(model_id: &str, variant: OnnxVariant, hw: &Hardware) -> Result<ModelProfile> {
    let files = resolve_model_files(model_id, variant, hw)?;
    let cfg: ModelConfig = read_json(&files.config_path)?;

    let tok_cfg_limit: usize = files
        .tokenizer_config_path
        .as_deref()
        .and_then(|p| read_json::<TokenizerCfg>(p).ok())
        .and_then(|t| t.effective_max_length())
        .unwrap_or(usize::MAX);

    let max_length = cfg.max_position_embeddings.min(tok_cfg_limit);

    let source = if tok_cfg_limit < cfg.max_position_embeddings {
        "tokenizer_config"
    } else {
        "config.max_position_embeddings"
    };
    tracing::info!(
        max_length,
        source,
        config_limit = cfg.max_position_embeddings,
        tok_cfg_limit,
        "resolved max_length",
    );

    let onnx_dir = files.onnx_path.parent().unwrap_or(files.config_path.parent().unwrap_or(Path::new(".")));
    let pooling = read_pooling_from_metadata(onnx_dir);
    let query_prefix = read_query_prefix_from_metadata(onnx_dir);

    Ok(ModelProfile {
        model_id: model_id.to_string(),
        onnx_path: files.onnx_path,
        tokenizer_path: files.tokenizer_path,
        dim: 0,
        max_length,
        pooling,
        query_prefix,
        hidden_size: cfg.hidden_size,
        num_hidden_layers: cfg.num_hidden_layers,
        intermediate_size: cfg.ffn_size(),
        num_attention_heads: cfg.num_attention_heads,
    })
}

pub fn resolve_model_files(model_id: &str, variant: OnnxVariant, hw: &Hardware) -> Result<ResolvedModel> {
    let p = Path::new(model_id);
    if p.exists() {
        resolve_local(p, variant, hw)
    } else {
        resolve_hf(model_id, variant, hw)
    }
}

fn resolve_local(p: &Path, variant: OnnxVariant, hw: &Hardware) -> Result<ResolvedModel> {
    let (dir, explicit_onnx) = if p.is_dir() {
        (p, None)
    } else if p.extension().and_then(|s| s.to_str()) == Some("onnx") {
        let parent = p
            .parent()
            .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", p.display()))?;
        (parent, Some(p))
    } else {
        anyhow::bail!("{} is neither a directory nor a .onnx file", p.display());
    };

    let onnx_path = match explicit_onnx {
        Some(file) => file.to_path_buf(),
        None => find_onnx_in(dir, variant, hw)?,
    };
    let tokenizer_path = find_tokenizer_in(dir)?;

    let config_path = dir.join("config.json");
    if !config_path.exists() {
        anyhow::bail!(
            "missing config.json in {} (the HF clone step should have placed it there)",
            dir.display()
        );
    }

    let tok_cfg = dir.join("tokenizer_config.json");
    let tokenizer_config_path = if tok_cfg.exists() { Some(tok_cfg) } else { None };

    tracing::info!(
        onnx = %onnx_path.display(),
        tokenizer = %tokenizer_path.display(),
        config = %config_path.display(),
        "using local model files"
    );

    Ok(ResolvedModel {
        onnx_path,
        tokenizer_path,
        config_path,
        tokenizer_config_path,
    })
}

fn find_onnx_in(dir: &Path, variant: OnnxVariant, hw: &Hardware) -> Result<PathBuf> {
    let one;
    let order: &[OnnxVariant] = match variant {
        OnnxVariant::Auto => auto_variant_order(hw),
        v => {
            one = [v];
            &one
        }
    };

    let mut best_rank: usize = usize::MAX;
    let mut best_path: Option<PathBuf> = None;

    let mut unknown_path: Option<PathBuf> = None;
    let mut unknown_count: usize = 0;

    let mut lower = String::with_capacity(64);

    for sub in ["", "onnx"] {
        let joined;
        let scan_dir: &Path = if sub.is_empty() {
            dir
        } else {
            joined = dir.join(sub);
            &joined
        };
        if !scan_dir.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(scan_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("onnx") {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };

            lower.clear();
            lower.push_str(name);
            lower.make_ascii_lowercase();

            match classify(&lower) {
                Some(v) => {
                    tracing::debug!(file = name, variant = ?v, "discovered onnx");
                    if let Some(rank) = order.iter().position(|cv| *cv == v)
                        && rank < best_rank
                    {
                        best_rank = rank;
                        best_path = Some(path);
                    }
                }
                None => {
                    unknown_count += 1;
                    if unknown_path.is_none() {
                        unknown_path = Some(path);
                    }
                }
            }
        }
    }

    if let Some(p) = best_path {
        tracing::info!(path = %p.display(), "selected ONNX variant");
        return Ok(p);
    }

    if variant != OnnxVariant::Auto {
        anyhow::bail!(
            "requested ONNX variant {} not available in {}",
            variant,
            dir.display(),
        );
    }

    if unknown_count == 1
        && let Some(p) = unknown_path
    {
        tracing::warn!(
            path = %p.display(),
            "no classified variant; using lone unrecognised .onnx",
        );
        return Ok(p);
    }
    if unknown_count > 1 {
        anyhow::bail!(
            "multiple unrecognised .onnx files in {} — pass --variant or a \
             direct .onnx path",
            dir.display(),
        );
    }
    anyhow::bail!("no .onnx file found in {}", dir.display())
}

fn find_tokenizer_in(dir: &Path) -> Result<PathBuf> {
    for c in TOKENIZER_CANDIDATES {
        let p = dir.join(c);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "no tokenizer.json found in {} (download it from the model's HF repo, \
         e.g. https://huggingface.co/<owner>/<name>/resolve/main/tokenizer.json)",
        dir.display()
    )
}

fn split_model_id(model_id: &str) -> Result<(&str, &str)> {
    let mut parts = model_id.splitn(2, '/');
    match (parts.next(), parts.next()) {
        (Some(o), Some(n)) if !o.is_empty() && !n.is_empty() => Ok((o, n)),
        _ => anyhow::bail!(
            "invalid model_id: expected 'owner/name', got '{}'",
            model_id
        ),
    }
}

fn resolve_hf(model_id: &str, variant: OnnxVariant, hw: &Hardware) -> Result<ResolvedModel> {
    let (owner, name) = split_model_id(model_id)?;

    let model_dir = zti_common::paths::models_dir()?.join(model_id.replace('/', "_"));
    std::fs::create_dir_all(&model_dir)?;

    let marker = model_dir.join(".zti_clone_complete");

    let need_clone = if marker.exists() {
        let local_sha = std::fs::read_to_string(&marker)
            .ok()
            .filter(|s| !s.trim().is_empty());
        match local_sha {
            Some(local_sha) => {
                let client = hf_hub::HFClientSync::new()?;
                let repo = client.model(owner, name);
                match repo.info().send() {
                    Ok(info) => match info.sha {
                        Some(ref remote_sha) if remote_sha != &local_sha => {
                            tracing::info!(
                                local = %local_sha,
                                remote = %remote_sha,
                                "HF revision changed, re-cloning"
                            );
                            true
                        }
                        _ => false,
                    },
                    Err(e) => {
                        tracing::debug!(error = %e, "skip SHA check (offline?)");
                        false
                    }
                }
            }
            None => false,
        }
    } else {
        !(model_dir.join("config.json").exists()
            && find_tokenizer_in(&model_dir).is_ok()
            && find_onnx_in(&model_dir, variant, hw).is_ok())
    };

    if need_clone {
        let client = hf_hub::HFClientSync::new()?;
        let repo = client.model(owner, name);

        let info = repo
            .info()
            .send()
            .with_context(|| format!("fetching HF repo info for {}", model_id))?;
        let sha = info.sha.clone().unwrap_or_default();

        tracing::info!(model = model_id, sha = %sha, "cloning HF repo (full)");

        let sha_opt = if sha.is_empty() {
            None
        } else {
            Some(sha.clone())
        };
        repo.snapshot_download()
            .maybe_revision(sha_opt)
            .local_dir(model_dir.clone())
            .send()
            .with_context(|| format!("downloading HF repo for {}", model_id))?;

        let bytes: u64 = walkdir::WalkDir::new(&model_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        tracing::info!(
            model = model_id,
            size_mb = bytes >> 20,
            dir = %model_dir.display(),
            "HF repo cloned",
        );

        let marker_content = if sha.is_empty() {
            "unknown"
        } else {
            &sha
        };
        std::fs::write(&marker, marker_content.as_bytes())
            .with_context(|| format!("writing {}", marker.display()))?;
    } else {
        tracing::debug!(model = model_id, "HF repo already cloned, skipping");
    }

    resolve_local(&model_dir, variant, hw)
}
