use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ort::session::Session;
use ort::session::SessionOutputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{TensorRef, ValueType};

use crate::batch::{BATCH_BUCKETS, BATCH_CEILING, SEQ_BUCKETS, TYPICAL_SEQ_LEN, next_bucket};
use crate::model_registry::{ModelProfile, OnnxVariant, resolve_profile};
use crate::normalize::normalize_l2;
use crate::pooling::{PoolingStrategy, pool_row_into};
use crate::tokenizer::{Tokenized, Tokenizer};
use zti_hw::{Device, EpStatus, Hardware, probe, register};

/// Flat pooled embeddings. `data` is laid out as `batch` contiguous rows of
/// `dim` `f32`s, so callers can hand `data` straight to an Arrow
/// `Float32Array` (or any other consumer that wants one big buffer) without a
/// per-row copy.
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
}

pub fn apply_prefix<'a>(text: &'a str, prefix: &Option<String>) -> Cow<'a, str> {
    match prefix {
        Some(p) => Cow::Owned(format!("{p}{text}")),
        None => Cow::Borrowed(text),
    }
}

static ONNX_AUTO: OnnxVariant = OnnxVariant::Auto;

#[derive(Debug, Clone)]
pub struct LoadOverrides<'a> {
    pub variant: &'a OnnxVariant,
    pub query_prefix: Option<&'a str>,
    pub passage_prefix: Option<&'a str>,
}

impl Default for LoadOverrides<'_> {
    fn default() -> Self {
        Self {
            variant: &ONNX_AUTO,
            query_prefix: None,
            passage_prefix: None,
        }
    }
}

/// Per-engine reusable buffers. Sized to the worst-case shape once at load;
/// every `embed_batch_*` call resizes (not reallocates) for the current
/// bucketed shape, keeping the hot path allocation-free.
struct Scratch {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
    valid_counts: Vec<usize>,
    pooled_flat: Vec<f32>,
}

impl Scratch {
    fn with_capacity(max_batch: usize, max_seq: usize, dim: usize, needs_tt: bool) -> Self {
        let total = max_batch * max_seq;
        Self {
            input_ids: Vec::with_capacity(total),
            attention_mask: Vec::with_capacity(total),
            token_type_ids: if needs_tt {
                Vec::with_capacity(total)
            } else {
                Vec::with_capacity(0)
            },
            valid_counts: Vec::with_capacity(max_batch),
            pooled_flat: Vec::with_capacity(max_batch * dim),
        }
    }

    fn prepare(&mut self, batch: usize, seq: usize, needs_tt: bool) {
        let total = batch * seq;
        self.input_ids.clear();
        self.input_ids.resize(total, 0);
        self.attention_mask.clear();
        self.attention_mask.resize(total, 0);
        if needs_tt {
            self.token_type_ids.clear();
            self.token_type_ids.resize(total, 0);
        }
        self.valid_counts.clear();
        self.valid_counts.resize(batch, 0);
    }
}

/// `Session` + `Scratch` live together under a single `std::sync::Mutex`.
/// The previous design held an async mutex across the synchronous ORT FFI
/// inside `session.run`, which starved the Tokio reactor. `block_in_place`
/// (used by the async wrappers) lets the worker thread yield back to the
/// runtime while we hold a sync mutex.
struct State {
    session: Session,
    scratch: Scratch,
}

pub struct EmbedEngine {
    state: Arc<Mutex<State>>,
    tokenizer: Tokenizer,
    profile: ModelProfile,
    hardware: Hardware,
    needs_token_type_ids: bool,
    warmed_shapes: Mutex<HashSet<(usize, usize)>>,
}

impl EmbedEngine {
    pub fn load(model_id: &str) -> Result<Self> {
        let hw = probe();
        tracing::info!(device = ?hw.device, cpus = hw.cpus, "probing hardware");
        Self::load_with(model_id, &hw, &LoadOverrides::default())
    }

    pub fn load_with_device(model_id: &str, hw: &Hardware) -> Result<Self> {
        Self::load_with(model_id, hw, &LoadOverrides::default())
    }

    pub fn load_with(model_id: &str, hw: &Hardware, opts: &LoadOverrides<'_>) -> Result<Self> {
        let mut profile = resolve_profile(
            model_id,
            opts.variant,
            hw,
            opts.query_prefix,
            opts.passage_prefix,
        )?;

        tracing::info!(path = %profile.onnx_path.display(), "loading ONNX model");

        // CoreML owns its own threading. Letting ORT spawn `hw.cpus` intra-op
        // threads just burns cores when CoreML pulls ops out of the CPU EP.
        let intra = match hw.device {
            Device::Metal => 1,
            _ => hw.cpus,
        };
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {}", e))?
            .with_intra_threads(intra)
            .map_err(|e| anyhow::anyhow!("intra threads: {}", e))?
            .with_execution_providers(register())
            .map_err(|e| anyhow::anyhow!("execution providers: {}", e))?;

        let mut session = builder.commit_from_file(&profile.onnx_path)?;
        let tokenizer = Tokenizer::from_file(&profile.tokenizer_path)?;

        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        if let Some(last_dim) = session
            .outputs()
            .first()
            .and_then(|o| match o.dtype() {
                ValueType::Tensor { shape, .. } => shape.last().copied(),
                _ => None,
            })
            .filter(|d| *d > 0)
        {
            profile.dim = last_dim as usize;
        }
        if let Some(tok_limit) = tokenizer.truncation_max_length() {
            profile.max_length = profile.max_length.min(tok_limit);
        }
        if profile.dim == 0 {
            let probed = probe_dim(&mut session, &profile, needs_token_type_ids, &tokenizer)?;
            if probed == 0 {
                anyhow::bail!("could not determine embedding dimension for {}", model_id);
            }
            profile.dim = probed;
            tracing::info!(dim = probed, "probed embedding dim by warmup");
        }

        let golden_seq = TYPICAL_SEQ_LEN.min(profile.max_length);
        let golden_batch = 8usize;
        let mut warmed_shapes: HashSet<(usize, usize)> = HashSet::new();
        let hw = hw.clone();
        if matches!(hw.device, Device::Metal) {
            let started = std::time::Instant::now();
            tracing::info!(
                batch = golden_batch,
                seq = golden_seq,
                "warming CoreML golden path (other shapes compile lazily on first hit)",
            );
            warmup_one(
                &mut session,
                &profile,
                needs_token_type_ids,
                golden_batch,
                golden_seq,
            )?;
            tracing::info!(
                ms = started.elapsed().as_millis() as u64,
                "CoreML golden path warm",
            );
            warmed_shapes.insert((golden_batch, golden_seq));

            let verify = std::time::Instant::now();
            warmup_one(
                &mut session,
                &profile,
                needs_token_type_ids,
                golden_batch,
                golden_seq,
            )?;
            let verify_ms = verify.elapsed().as_millis() as u64;

            let ep_threshold_ms = 500u64;
            if verify_ms > ep_threshold_ms {
                tracing::warn!(
                    verify_ms,
                    ep_threshold_ms,
                    "CoreML may not be using GPU/ANE — inference slower than expected. \
                     Consider --variant fp32 for GPU acceleration.",
                );
                hw.ep_status.set(EpStatus::Fallback);
            } else {
                tracing::info!(verify_ms, "CoreML GPU/ANE acceleration verified");
                hw.ep_status.set(EpStatus::Active);
            }
        }

        tracing::info!(
            dim = profile.dim,
            max_len = profile.max_length,
            model = %model_id,
            "embed engine ready"
        );

        let scratch = Scratch::with_capacity(
            BATCH_CEILING,
            profile.max_length,
            profile.dim,
            needs_token_type_ids,
        );

        Ok(Self {
            state: Arc::new(Mutex::new(State { session, scratch })),
            tokenizer,
            profile,
            hardware: hw,
            needs_token_type_ids,
            warmed_shapes: Mutex::new(warmed_shapes),
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

    pub fn recommended_batch_size(&self) -> usize {
        crate::batch::recommended_batch_size(&self.profile, &self.hardware)
    }

    /// Tokenize a batch once. Callers sort the result by `ids.len()` to feed
    /// length-homogeneous batches into [`Self::embed_batch_tokenized_async`],
    /// which keeps bucketed padding tight without re-running the tokenizer.
    pub fn tokenize(&self, texts: &[&str]) -> Result<Vec<Tokenized>> {
        self.tokenizer.encode_batch(texts)
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encs = self.tokenizer.encode_batch(texts)?;
        let refs: Vec<&Tokenized> = encs.iter().collect();
        let pooled = self.embed_batch_tokenized_sync(&refs)?;
        Ok(pooled_to_vec_of_vecs(pooled))
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encs = self.tokenizer.encode_batch(texts)?;
        let refs: Vec<&Tokenized> = encs.iter().collect();
        let pooled = tokio::task::block_in_place(|| self.embed_batch_tokenized_sync(&refs))?;
        Ok(pooled_to_vec_of_vecs(pooled))
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, &self.profile.query_prefix);
        let mut batch = self.embed_batch(&[&input])?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no embedding produced"))
    }

    pub async fn embed_query_async(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, &self.profile.query_prefix);
        let mut batch = self.embed_batch_async(&[&input]).await?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no embedding produced"))
    }

    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, &self.profile.passage_prefix);
        let mut batch = self.embed_batch(&[&input])?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no embedding produced"))
    }

    pub async fn embed_passage_async(&self, text: &str) -> Result<Vec<f32>> {
        let input = apply_prefix(text, &self.profile.passage_prefix);
        let mut batch = self.embed_batch_async(&[&input]).await?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no embedding produced"))
    }

    /// Embed a pre-tokenized batch. The slice carries borrowed [`Tokenized`]
    /// values so the caller can compose a batch out of any (possibly
    /// non-contiguous) subset of an already-tokenized set without cloning.
    /// Uses `tokio::task::block_in_place` to execute the synchronous ORT FFI
    /// while letting the Tokio worker yield to the runtime — this requires a
    /// multi-threaded runtime (the workspace enables `rt-multi-thread`).
    pub async fn embed_batch_tokenized_async(&self, encs: &[&Tokenized]) -> Result<Pooled> {
        tokio::task::block_in_place(|| self.embed_batch_tokenized_sync(encs))
    }

    pub fn embed_batch_tokenized_sync(&self, encs: &[&Tokenized]) -> Result<Pooled> {
        let real_batch = encs.len();
        if real_batch == 0 {
            return Ok(Pooled::empty(self.profile.dim));
        }
        if real_batch > BATCH_CEILING {
            anyhow::bail!(
                "batch size {} exceeds BATCH_CEILING ({})",
                real_batch,
                BATCH_CEILING,
            );
        }

        let max_len = self.profile.max_length;
        let real_seq = encs
            .iter()
            .map(|e| e.ids.len().min(max_len))
            .max()
            .unwrap_or(1)
            .max(1);
        let seq = next_bucket(SEQ_BUCKETS, real_seq, max_len);
        let batch = next_bucket(BATCH_BUCKETS, real_batch, BATCH_CEILING);

        self.note_shape_seen(batch, seq);

        let mut state = self.state.lock().expect("embed state poisoned");
        let State { session, scratch } = &mut *state;
        scratch.prepare(batch, seq, self.needs_token_type_ids);
        fill_scratch(scratch, encs, seq);

        run_and_pool(
            session,
            scratch,
            Shape {
                batch,
                seq,
                real_batch,
                dim: self.profile.dim,
            },
            self.needs_token_type_ids,
            &PoolingStrategy::from(self.profile.pooling),
        )?;

        // One allocation total: a flat `batch * dim` f32 buffer. Beats the
        // old `Vec<Vec<f32>>` path which paid an alloc per row, then made the
        // indexer re-extend everything into a flat Arrow buffer anyway.
        Ok(Pooled {
            data: scratch.pooled_flat.clone(),
            dim: self.profile.dim,
            batch: real_batch,
        })
    }

    /// First time a (batch, seq) shape is seen post-warmup on Metal, the
    /// CoreML EP JIT-compiles a fresh MLProgram (cached to disk for next
    /// run). Surface that one-time stall so users understand the pause.
    fn note_shape_seen(&self, batch: usize, seq: usize) {
        if !matches!(self.hardware.device, Device::Metal) {
            return;
        }
        let mut seen = self.warmed_shapes.lock().expect("warmed_shapes poisoned");
        if seen.insert((batch, seq)) && seen.len() > 1 {
            tracing::info!(
                batch,
                seq,
                "CoreML JIT-compiling new shape (one-time stall, 1–3 s; cached on disk)",
            );
        }
    }
}

fn pooled_to_vec_of_vecs(p: Pooled) -> Vec<Vec<f32>> {
    if p.dim == 0 {
        return Vec::new();
    }
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(p.batch);
    for chunk in p.data.chunks(p.dim) {
        out.push(chunk.to_vec());
    }
    out
}

fn fill_scratch(scratch: &mut Scratch, encs: &[&Tokenized], seq: usize) {
    for (i, tok) in encs.iter().enumerate() {
        let len = tok.ids.len().min(seq);
        scratch.valid_counts[i] = len;
        let base = i * seq;
        let dst_ids = &mut scratch.input_ids[base..base + len];
        let dst_mask = &mut scratch.attention_mask[base..base + len];
        let src_ids = &tok.ids[..len];
        let src_mask = &tok.mask[..len];
        for (d, &s) in dst_ids.iter_mut().zip(src_ids) {
            *d = s as i64;
        }
        for (d, &s) in dst_mask.iter_mut().zip(src_mask) {
            *d = s as i64;
        }
    }
    // Rows i in [real_batch..batch) stay all-zero with valid_counts[i] == 0;
    // extract_pooled_into iterates only up to real_batch, so the padding
    // rows never reach the caller.
}

/// One bucketed inference: `batch` × `seq` are the padded shape fed to the
/// session; `real_batch` and `dim` describe the meaningful output rows.
#[derive(Debug, Clone, Copy)]
struct Shape {
    batch: usize,
    seq: usize,
    real_batch: usize,
    dim: usize,
}

fn run_and_pool(
    session: &mut Session,
    scratch: &mut Scratch,
    shape: Shape,
    needs_tt: bool,
    strategy: &PoolingStrategy,
) -> Result<()> {
    let Shape {
        batch,
        seq,
        real_batch,
        dim,
    } = shape;
    let ids_t = TensorRef::from_array_view(([batch, seq], scratch.input_ids.as_slice()))?;
    let mask_t = TensorRef::from_array_view(([batch, seq], scratch.attention_mask.as_slice()))?;
    let mut inputs = ort::inputs![
        "input_ids" => ids_t,
        "attention_mask" => mask_t,
    ];
    if needs_tt {
        let tt_t = TensorRef::from_array_view(([batch, seq], scratch.token_type_ids.as_slice()))?;
        inputs.extend(ort::inputs!["token_type_ids" => tt_t]);
    }

    let outputs = session.run(inputs)?;
    extract_pooled_into(
        &outputs,
        seq,
        real_batch,
        dim,
        &scratch.valid_counts,
        strategy,
        &mut scratch.pooled_flat,
    )
}

fn extract_pooled_into(
    outputs: &SessionOutputs<'_>,
    seq: usize,
    real_batch: usize,
    dim: usize,
    valid_counts: &[usize],
    strategy: &PoolingStrategy,
    pooled_flat: &mut Vec<f32>,
) -> Result<()> {
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let stride = seq * dim;
    pooled_flat.clear();
    pooled_flat.resize(real_batch * dim, 0.0);
    for i in 0..real_batch {
        let row_start = i * stride;
        let row_data = &data[row_start..row_start + stride];
        let out = &mut pooled_flat[i * dim..(i + 1) * dim];
        pool_row_into(strategy, row_data, valid_counts[i], out);
        normalize_l2(out);
    }
    Ok(())
}

fn warmup_one(
    session: &mut Session,
    profile: &ModelProfile,
    needs_tt: bool,
    batch: usize,
    seq: usize,
) -> Result<()> {
    let dim = profile.dim.max(1);
    let mut scratch = Scratch::with_capacity(batch, seq, dim, needs_tt);
    scratch.prepare(batch, seq, needs_tt);
    // Mask = 1 on every position so attention has at least one valid token to
    // attend to. The model output is discarded — we only need to force the
    // CoreML EP to compile the MLProgram for this shape.
    for v in scratch.attention_mask.iter_mut() {
        *v = 1;
    }
    let ids_t = TensorRef::from_array_view(([batch, seq], scratch.input_ids.as_slice()))?;
    let mask_t = TensorRef::from_array_view(([batch, seq], scratch.attention_mask.as_slice()))?;
    let mut inputs = ort::inputs![
        "input_ids" => ids_t,
        "attention_mask" => mask_t,
    ];
    if needs_tt {
        let tt_t = TensorRef::from_array_view(([batch, seq], scratch.token_type_ids.as_slice()))?;
        inputs.extend(ort::inputs!["token_type_ids" => tt_t]);
    }
    let _ = session.run(inputs)?;
    Ok(())
}

fn probe_dim(
    session: &mut Session,
    profile: &ModelProfile,
    needs_tt: bool,
    tokenizer: &Tokenizer,
) -> Result<usize> {
    let enc = tokenizer.encode("a")?;
    if enc.ids.is_empty() {
        anyhow::bail!("warmup produced no tokens");
    }
    let real_seq = enc.ids.len().min(profile.max_length).max(1);
    let seq = next_bucket(SEQ_BUCKETS, real_seq, profile.max_length);
    let batch = 1usize;

    let mut scratch = Scratch::with_capacity(batch, seq, 1, needs_tt);
    scratch.prepare(batch, seq, needs_tt);
    let len = enc.ids.len().min(seq);
    scratch.valid_counts[0] = len;
    for (d, &s) in scratch.input_ids[..len].iter_mut().zip(&enc.ids) {
        *d = s as i64;
    }
    for (d, &s) in scratch.attention_mask[..len].iter_mut().zip(&enc.mask) {
        *d = s as i64;
    }

    let ids_t = TensorRef::from_array_view(([batch, seq], scratch.input_ids.as_slice()))?;
    let mask_t = TensorRef::from_array_view(([batch, seq], scratch.attention_mask.as_slice()))?;
    let mut inputs = ort::inputs![
        "input_ids" => ids_t,
        "attention_mask" => mask_t,
    ];
    if needs_tt {
        let tt_t = TensorRef::from_array_view(([batch, seq], scratch.token_type_ids.as_slice()))?;
        inputs.extend(ort::inputs!["token_type_ids" => tt_t]);
    }
    let outputs = session.run(inputs)?;
    let (shape, _data) = outputs[0].try_extract_tensor::<f32>()?;
    Ok(shape.last().copied().unwrap_or(0) as usize)
}
