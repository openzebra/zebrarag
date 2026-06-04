use std::borrow::Cow;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use arrow::array::{FixedSizeListArray, Float32Array};
use arrow::datatypes::{DataType, Field};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::Config as BertConfig;
use candle_transformers::models::jina_bert::Config as JinaConfig;
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};

use crate::bert::BertModel;
use crate::jina_bert::JinaBertModel;

use crate::batch::{BATCH_BUCKETS, BATCH_CEILING, SEQ_BUCKETS, next_bucket};
use crate::model_registry::{ModelProfile, PoolingStrategyEnum, read_json, resolve_profile};
use crate::normalize::normalize_l2;
use crate::pooling::{PoolingStrategy, pool_row_into};
use crate::tokenizer::{Tokenized, Tokenizer};
use zti_hw::{Hardware, candle_device, probe};

#[derive(Debug, Clone)]
pub struct Pooled {
    pub data: Vec<f32>,
    pub dim: usize,
    pub batch: usize,
}

impl Pooled {
    pub fn empty(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            batch: 0,
        }
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks(self.dim)
    }

    pub fn into_float32_array(self) -> Float32Array {
        Float32Array::from(self.data)
    }

    pub fn into_fixed_size_list(self) -> FixedSizeListArray {
        FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            self.dim as i32,
            Arc::new(Float32Array::from(self.data)),
            None,
        )
    }
}

pub fn apply_prefix<'a>(text: &'a str, prefix: Option<&str>) -> Cow<'a, str> {
    match prefix {
        Some(p) => Cow::Owned(format!("{p}{text}")),
        None => Cow::Borrowed(text),
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadOverrides<'a> {
    pub query_prefix: Option<&'a str>,
    pub passage_prefix: Option<&'a str>,
    pub model_dtype: Option<DType>,
}

pub fn parse_model_dtype(raw: &str) -> Option<DType> {
    match raw.to_ascii_lowercase().as_str() {
        "f16" | "float16" | "half" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        "f32" | "float32" | "float" => Some(DType::F32),
        _ => None,
    }
}

struct Scratch {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    valid_counts: Vec<usize>,
    pooled_flat: Vec<f32>,
}

impl Scratch {
    fn with_capacity(max_batch: usize, max_seq: usize, dim: usize) -> Self {
        let total = max_batch * max_seq;
        Self {
            input_ids: Vec::with_capacity(total),
            attention_mask: Vec::with_capacity(total),
            valid_counts: Vec::with_capacity(max_batch),
            pooled_flat: Vec::with_capacity(max_batch * dim),
        }
    }

    fn prepare(&mut self, batch: usize, seq: usize) {
        let total = batch * seq;
        self.input_ids.clear();
        self.input_ids.resize(total, 0);
        self.attention_mask.clear();
        self.attention_mask.resize(total, 0);
        self.valid_counts.clear();
        self.valid_counts.resize(batch, 0);
    }
}

enum Model {
    Bert(BertModel),
    Jina(JinaBertModel),
}

impl Model {
    fn forward(
        &self,
        ids: &Tensor,
        token_type_ids: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Bert(model) => model.forward(ids, token_type_ids, mask),
            Self::Jina(model) => model.forward(ids, token_type_ids, mask),
        }
    }
}

struct State {
    model: Model,
    device: candle_core::Device,
    scratch: Scratch,
}

#[derive(Debug, Deserialize)]
struct PositionEmbeddingPeek {
    #[serde(default)]
    position_embedding_type: Option<String>,
}

/// Immutable shape/pooling config the worker needs per request. Captured once
/// at load (post-warmup) and owned by the worker thread.
#[derive(Clone, Copy)]
struct WorkerCfg {
    dim: usize,
    max_length: usize,
    pooling: PoolingStrategyEnum,
}

/// One embedding job: the shared tokenized batch, the indices into it to embed,
/// and a one-shot channel to return the pooled result.
struct EmbedRequest {
    encs: Arc<Vec<Tokenized>>,
    idxs: Arc<[usize]>,
    reply: oneshot::Sender<Result<Pooled>>,
}

pub struct EmbedEngine {
    /// Hands jobs to the single thread that owns the model/device/scratch. The
    /// reactor never blocks on GPU work and the model is never contended —
    /// exactly one thread ever runs a forward pass.
    tx: mpsc::UnboundedSender<EmbedRequest>,
    /// Clone source for [`Self::device`]. The model lives on the worker; this
    /// is a cheap, uncontended handle used only to build the GPU rerank scorer
    /// on the search path.
    device: Mutex<candle_core::Device>,
    tokenizer: Tokenizer,
    profile: ModelProfile,
    hardware: Arc<Hardware>,
}

impl EmbedEngine {
    pub fn load(model_id: &str) -> Result<Self> {
        let hw = Arc::new(probe());
        Self::load_with(model_id, hw, &LoadOverrides::default())
    }

    pub fn load_with_device(model_id: &str, hw: Arc<Hardware>) -> Result<Self> {
        Self::load_with(model_id, hw, &LoadOverrides::default())
    }

    pub fn load_with(model_id: &str, hw: Arc<Hardware>, opts: &LoadOverrides<'_>) -> Result<Self> {
        let mut profile = resolve_profile(model_id, opts.query_prefix, opts.passage_prefix)?;

        tracing::info!(path = %profile.weights_path.display(), "loading safetensors model");

        let device = candle_device(&hw);
        let dtype = opts.model_dtype.unwrap_or(DType::F32);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&profile.weights_path], dtype, &device)?
        };

        let model = match read_json::<PositionEmbeddingPeek>(&profile.config_path)?
            .position_embedding_type
            .as_deref()
        {
            Some("alibi") => {
                let config: JinaConfig = read_json(&profile.config_path)?;
                Model::Jina(JinaBertModel::load(vb, &config)?)
            }
            _ => {
                let config: BertConfig = read_json(&profile.config_path)?;
                Model::Bert(BertModel::load(vb, &config)?)
            }
        };
        let tokenizer = Tokenizer::from_file(&profile.tokenizer_path)?;

        if let Some(tok_limit) = tokenizer.truncation_max_length() {
            profile.max_length = profile.max_length.min(tok_limit);
        }

        // Warmup forward: validates the model is NaN-free for this dtype/device
        // and probes the embedding dim when the config didn't provide it. A
        // reduced-precision dtype that overflows (e.g. a candle mask bug) is
        // rejected here instead of silently wasting an entire index.
        {
            let enc = tokenizer.encode("a")?;
            if enc.ids.is_empty() {
                anyhow::bail!("warmup produced no tokens");
            }
            let ids = Tensor::from_slice(&enc.ids, (1, enc.ids.len()), &device)?;
            let token_type_ids = Tensor::zeros_like(&ids)?;
            let mask = Tensor::ones_like(&ids)?;
            let out = model.forward(&ids, &token_type_ids, Some(&mask))?;
            if profile.dim == 0 {
                profile.dim = out.dims()[2];
                tracing::info!(dim = profile.dim, "probed embedding dim");
            }
            let flat = out
                .to_device(&candle_core::Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if flat.iter().any(|v| v.is_nan()) {
                anyhow::bail!(
                    "model produced NaN at load (dtype {dtype:?}, device {:?}) — \
                     this precision is unsupported for this model/device",
                    hw.device
                );
            }
        }

        tracing::info!(
            dim = profile.dim,
            max_len = profile.max_length,
            device = ?hw.device,
            model = %model_id,
            "embed engine ready"
        );

        let scratch = Scratch::with_capacity(BATCH_CEILING, profile.max_length, profile.dim);

        // Spawn the single embedding worker. It owns the model, device, and
        // scratch for its whole life; callers submit jobs over `tx` and await a
        // one-shot reply. This keeps GPU work off the tokio reactor (no
        // `block_in_place`) and removes the global model mutex — only this
        // thread ever runs a forward pass.
        let device_handle = device.clone();
        let cfg = WorkerCfg {
            dim: profile.dim,
            max_length: profile.max_length,
            pooling: profile.pooling,
        };
        let (tx, mut rx) = mpsc::unbounded_channel::<EmbedRequest>();
        let mut state = State {
            model,
            device,
            scratch,
        };
        std::thread::Builder::new()
            .name("zti-embed".into())
            .spawn(move || {
                while let Some(req) = rx.blocking_recv() {
                    let refs: Vec<&Tokenized> =
                        req.idxs.iter().filter_map(|&i| req.encs.get(i)).collect();
                    let _ = req.reply.send(embed_on_state(&mut state, &refs, &cfg));
                }
            })
            .map_err(|e| anyhow!("spawn embed worker: {e}"))?;

        Ok(Self {
            tx,
            device: Mutex::new(device_handle),
            tokenizer,
            profile,
            hardware: hw,
        })
    }

    pub fn dim(&self) -> usize {
        self.profile.dim
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn hardware(&self) -> &Hardware {
        &self.hardware
    }

    pub fn device(&self) -> Result<candle_core::Device> {
        let device = self
            .device
            .lock()
            .map_err(|_| anyhow!("embed device lock poisoned"))?;
        Ok(device.clone())
    }

    pub fn recommended_batch_size(&self) -> usize {
        crate::batch::recommended_batch_size(&self.profile, &self.hardware)
    }

    pub fn tokenize(&self, texts: &[&str]) -> Result<Vec<Tokenized>> {
        self.tokenizer.encode_batch(texts)
    }

    /// Count tokens in a single text without retaining ids/mask buffers.
    #[inline]
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        self.tokenizer.count_tokens(text)
    }

    /// True when the tokenizer truncates (token counts would be capped → unreliable bpt).
    #[inline]
    pub fn truncates(&self) -> bool {
        self.tokenizer.truncation_max_length().is_some()
    }

    /// Queue a tokenized batch on the embedding worker and return immediately.
    /// Await the returned receiver to collect the pooled result. `idxs` selects
    /// rows of `encs`; the heavy token buffers stay shared via `Arc`, and only
    /// the small index list is owned per request.
    pub fn submit(
        &self,
        encs: Arc<Vec<Tokenized>>,
        idxs: Arc<[usize]>,
    ) -> Result<oneshot::Receiver<Result<Pooled>>> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EmbedRequest { encs, idxs, reply })
            .map_err(|_| anyhow!("embed worker thread is gone"))?;
        Ok(rx)
    }

    /// Submit a tokenized batch to the embedding worker and await the pooled
    /// result. The reactor never blocks: the GPU forward runs on the worker
    /// thread while this task is parked on the one-shot.
    pub async fn embed_tokenized(
        &self,
        encs: Arc<Vec<Tokenized>>,
        idxs: Vec<usize>,
    ) -> Result<Pooled> {
        self.submit(encs, idxs.into())?
            .await
            .map_err(|_| anyhow!("embed worker dropped without replying"))?
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encs = self.tokenizer.encode_batch(texts)?;
        let n = encs.len();
        let pooled = self
            .embed_tokenized(Arc::new(encs), (0..n).collect())
            .await?;
        Ok(pooled.rows().map(<[f32]>::to_vec).collect())
    }

    pub async fn embed_query_async(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, self.profile.query_prefix.as_deref());
        let encs = self.tokenizer.encode_batch(&[&*input])?;
        let n = encs.len();
        let pooled = self
            .embed_tokenized(Arc::new(encs), (0..n).collect())
            .await?;
        if pooled.data.is_empty() {
            anyhow::bail!("no embedding produced");
        }
        Ok(pooled.data)
    }

    pub async fn embed_passage_async(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, self.profile.passage_prefix.as_deref());
        let encs = self.tokenizer.encode_batch(&[&*input])?;
        let n = encs.len();
        let pooled = self
            .embed_tokenized(Arc::new(encs), (0..n).collect())
            .await?;
        if pooled.data.is_empty() {
            anyhow::bail!("no embedding produced");
        }
        Ok(pooled.data)
    }
}

/// Run one embedding job against the worker-owned `State`. Numerically
/// identical to the previous in-mutex path — only the call site (a dedicated
/// thread instead of `block_in_place`) changed.
fn embed_on_state(state: &mut State, encs: &[&Tokenized], cfg: &WorkerCfg) -> Result<Pooled> {
    let real_batch = encs.len();
    if real_batch == 0 {
        return Ok(Pooled::empty(cfg.dim));
    }
    if real_batch > BATCH_CEILING {
        anyhow::bail!(
            "batch size {} exceeds BATCH_CEILING ({})",
            real_batch,
            BATCH_CEILING,
        );
    }

    let max_len = cfg.max_length;
    let real_seq = encs
        .iter()
        .map(|e| e.ids.len().min(max_len))
        .max()
        .unwrap_or(1)
        .max(1);
    let seq = next_bucket(SEQ_BUCKETS, real_seq, max_len);
    let batch = next_bucket(BATCH_BUCKETS, real_batch, BATCH_CEILING);

    let State {
        model,
        device,
        scratch,
    } = state;
    scratch.prepare(batch, seq);
    fill_scratch(scratch, encs, seq);

    run_and_pool(
        model,
        device,
        scratch,
        Shape {
            batch,
            seq,
            real_batch,
            dim: cfg.dim,
        },
        &PoolingStrategy::from(cfg.pooling),
    )?;

    Ok(Pooled {
        data: std::mem::take(&mut scratch.pooled_flat),
        dim: cfg.dim,
        batch: real_batch,
    })
}

fn fill_scratch(scratch: &mut Scratch, encs: &[&Tokenized], seq: usize) {
    for (i, tok) in encs.iter().enumerate() {
        let len = tok.ids.len().min(seq);
        scratch.valid_counts[i] = len;
        let base = i * seq;
        scratch.input_ids[base..base + len].copy_from_slice(&tok.ids[..len]);
        scratch.attention_mask[base..base + len].copy_from_slice(&tok.mask[..len]);
    }
}

#[derive(Debug, Clone, Copy)]
struct Shape {
    batch: usize,
    seq: usize,
    real_batch: usize,
    dim: usize,
}

fn run_and_pool(
    model: &Model,
    device: &candle_core::Device,
    scratch: &mut Scratch,
    shape: Shape,
    strategy: &PoolingStrategy,
) -> Result<()> {
    let Shape {
        batch,
        seq,
        real_batch,
        dim,
    } = shape;

    let ids = Tensor::from_slice(&scratch.input_ids[..batch * seq], (batch, seq), device)?;
    let token_type_ids = Tensor::zeros_like(&ids)?;
    let mask = Tensor::from_slice(&scratch.attention_mask[..batch * seq], (batch, seq), device)?;

    let output = model.forward(&ids, &token_type_ids, Some(&mask))?;
    let output = output
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let data_flat = output.flatten_all()?.to_vec1::<f32>()?;

    scratch.pooled_flat.clear();
    scratch.pooled_flat.resize(real_batch * dim, 0.0);

    for i in 0..real_batch {
        let start = i * seq * dim;
        let row_flat = &data_flat[start..start + seq * dim];
        let out = &mut scratch.pooled_flat[i * dim..(i + 1) * dim];
        pool_row_into(strategy, row_flat, scratch.valid_counts[i], out);
        normalize_l2(out);
    }

    Ok(())
}

#[cfg(test)]
mod worker_tests {
    use super::*;

    /// Loads the model named by `ZTI_TEST_MODEL`, or `None` (skip) when unset
    /// or unresolvable — clean CI has no weights, the daemon host does.
    fn test_engine() -> Option<EmbedEngine> {
        let model = std::env::var("ZTI_TEST_MODEL").ok()?;
        match EmbedEngine::load(&model) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("skipping worker test: cannot load {model}: {e}");
                None
            }
        }
    }

    /// The embed math is moved verbatim into `embed_on_state`, so equality with
    /// the prior sync path holds by construction. What the worker rewrite can
    /// still break is the *plumbing*: this asserts the worker is deterministic
    /// and that many concurrent jobs all resolve to the same vector with no
    /// deadlock or cross-request scratch races.
    #[test]
    fn worker_embed_is_deterministic_and_concurrent() {
        // Load the engine *outside* any runtime — exactly as the daemon does
        // (hf-hub blocks internally on load, which panics inside a runtime).
        let Some(engine) = test_engine() else {
            return;
        };
        let engine = Arc::new(engine);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("build test runtime");

        rt.block_on(async {
            let query = "fn parse_config(path: &str) -> Result<Config>";

            let base = engine.embed_query_async(query).await.expect("embed base");
            assert_eq!(base.len(), engine.dim(), "embedding width must equal dim");
            assert!(
                base.iter().any(|v| *v != 0.0),
                "embedding must be non-trivial"
            );

            let again = engine.embed_query_async(query).await.expect("embed again");
            assert_eq!(base, again, "worker embeddings must be deterministic");

            let mut handles = Vec::with_capacity(16);
            for _ in 0..16 {
                let e = Arc::clone(&engine);
                let q = query.to_string();
                handles.push(tokio::spawn(async move { e.embed_query_async(&q).await }));
            }
            for h in handles {
                let v = h.await.expect("join").expect("concurrent embed");
                assert_eq!(v, base, "concurrent embedding must equal the serial result");
            }
        });
    }
}
