use std::sync::Arc;

use anyhow::Result;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, SessionOutputs};
use ort::value::ValueType;
use tokio::sync::Mutex;

use crate::model_registry::{ModelProfile, OnnxVariant, resolve_profile};
use crate::normalize::normalize_l2;
use crate::pooling::{PoolingStrategy, pool_row};
use crate::tokenizer::{Tokenized, Tokenizer};
use zti_hw::{Hardware, probe, register};

#[derive(Debug, Clone, Copy)]
pub struct LoadOverrides<'a> {
    pub variant: OnnxVariant,
    pub query_prefix: Option<&'a str>,
}

impl<'a> Default for LoadOverrides<'a> {
    fn default() -> Self {
        Self {
            variant: OnnxVariant::Auto,
            query_prefix: None,
        }
    }
}

pub struct EmbedEngine {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    profile: ModelProfile,
    hardware: Hardware,
    needs_token_type_ids: bool,
}

struct Prepared {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids_i64: Option<Vec<i64>>,
    valid_counts: Vec<usize>,
    batch: usize,
    seq: usize,
    dim: usize,
    strategy: PoolingStrategy,
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
        let mut profile = resolve_profile(model_id, opts.variant, hw, opts.query_prefix)?;

        tracing::info!(path = %profile.onnx_path.display(), "loading ONNX model");

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {}", e))?
            .with_intra_threads(hw.cpus)
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
            // Shape was dynamic. Probe by running a single token through the
            // session and reading the resulting last-axis length. We have
            // exclusive access to `session` here, so we can call `run` directly
            // without a lock.
            let prep = prepare(&tokenizer, &profile, needs_token_type_ids, &["a"])?
                .ok_or_else(|| anyhow::anyhow!("warmup produced no tokens for {}", model_id))?;
            let probe = run_and_pool_sync(&mut session, prep)?;
            let probed = probe.first().map(|v| v.len()).unwrap_or(0);
            if probed == 0 {
                anyhow::bail!("could not determine embedding dimension for {}", model_id);
            }
            profile.dim = probed;
            tracing::info!(dim = probed, "probed embedding dim by warmup");
        }

        tracing::info!(
            dim = profile.dim,
            max_len = profile.max_length,
            model = %model_id,
            "embed engine ready"
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            profile,
            hardware: *hw,
            needs_token_type_ids,
        })
    }

    fn prepare(&self, texts: &[&str]) -> Result<Option<Prepared>> {
        prepare(
            &self.tokenizer,
            &self.profile,
            self.needs_token_type_ids,
            texts,
        )
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prep = match self.prepare(texts)? {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };
        let mut guard = self.session.blocking_lock();
        run_and_pool_sync(&mut guard, prep)
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prep = match self.prepare(texts)? {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };
        let mut guard = self.session.lock().await;
        run_and_pool_async(&mut guard, prep).await
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = match &self.profile.query_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        let mut batch = self.embed_batch(&[&input])?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no embedding produced"))
    }

    pub async fn embed_query_async(&self, text: &str) -> Result<Vec<f32>> {
        let input = match &self.profile.query_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        let mut batch = self.embed_batch_async(&[&input]).await?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no embedding produced"))
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
    /// length-homogeneous batches into `embed_batch_tokenized_async`, which
    /// keeps `prepare_from_encs`'s dynamic padding tight without re-running
    /// the tokenizer.
    pub fn tokenize(&self, texts: &[&str]) -> Result<Vec<Tokenized>> {
        self.tokenizer.encode_batch(texts)
    }

    /// Embed a pre-tokenized batch. The slice carries borrowed `Tokenized`
    /// values so the caller can compose a batch out of any (possibly
    /// non-contiguous) subset of an already-tokenized set without cloning
    /// or rearranging the backing storage.
    pub async fn embed_batch_tokenized_async(
        &self,
        encs: &[&Tokenized],
    ) -> Result<Vec<Vec<f32>>> {
        let prep = match prepare_from_encs(&self.profile, self.needs_token_type_ids, encs) {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };
        let mut guard = self.session.lock().await;
        run_and_pool_async(&mut guard, prep).await
    }
}

fn prepare(
    tokenizer: &Tokenizer,
    profile: &ModelProfile,
    needs_token_type_ids: bool,
    texts: &[&str],
) -> Result<Option<Prepared>> {
    if texts.is_empty() {
        return Ok(None);
    }
    let encs = tokenizer.encode_batch(texts)?;
    let mut refs: Vec<&Tokenized> = Vec::with_capacity(encs.len());
    refs.extend(encs.iter());
    Ok(prepare_from_encs(profile, needs_token_type_ids, &refs))
}

fn prepare_from_encs(
    profile: &ModelProfile,
    needs_token_type_ids: bool,
    encs: &[&Tokenized],
) -> Option<Prepared> {
    if encs.is_empty() {
        return None;
    }

    let max_len = profile.max_length;
    let batch = encs.len();
    let dim = profile.dim;

    let mut max_seq = 0usize;
    for enc in encs {
        let len = enc.ids.len().min(max_len);
        if len > max_seq {
            max_seq = len;
        }
    }
    if max_seq == 0 {
        max_seq = 1;
    }
    let seq = max_seq;

    let total = batch * seq;
    let mut input_ids = vec![0i64; total];
    let mut attention_mask = vec![0i64; total];
    let token_type_ids_i64 = needs_token_type_ids.then(|| vec![0i64; total]);
    let mut valid_counts = vec![0usize; batch];

    for (i, tok) in encs.iter().enumerate() {
        let len = tok.ids.len().min(seq);
        valid_counts[i] = len;
        let base = i * seq;
        let dst_ids = &mut input_ids[base..base + len];
        let dst_mask = &mut attention_mask[base..base + len];
        let src_ids = &tok.ids[..len];
        let src_mask = &tok.mask[..len];
        for (d, &s) in dst_ids.iter_mut().zip(src_ids) {
            *d = s as i64;
        }
        for (d, &s) in dst_mask.iter_mut().zip(src_mask) {
            *d = s as i64;
        }
    }

    Some(Prepared {
        input_ids,
        attention_mask,
        token_type_ids_i64,
        valid_counts,
        batch,
        seq,
        dim,
        strategy: PoolingStrategy::from(profile.pooling),
    })
}

fn run_and_pool_sync(session: &mut Session, prep: Prepared) -> Result<Vec<Vec<f32>>> {
    let Prepared {
        input_ids,
        attention_mask,
        token_type_ids_i64,
        valid_counts,
        batch,
        seq,
        dim,
        strategy,
    } = prep;

    let ids_tensor = ort::value::Tensor::from_array(([batch, seq], input_ids))?;
    let mask_tensor = ort::value::Tensor::from_array(([batch, seq], attention_mask))?;

    let mut inputs = ort::inputs![
        "input_ids" => ids_tensor,
        "attention_mask" => mask_tensor,
    ];

    if let Some(tt) = token_type_ids_i64 {
        let tt_tensor = ort::value::Tensor::from_array(([batch, seq], tt))?;
        inputs.extend(ort::inputs!["token_type_ids" => tt_tensor]);
    }

    let outputs = session.run(inputs)?;
    extract_pooled(&outputs, batch, seq, dim, &valid_counts, &strategy)
}

async fn run_and_pool_async(session: &mut Session, prep: Prepared) -> Result<Vec<Vec<f32>>> {
    let Prepared {
        input_ids,
        attention_mask,
        token_type_ids_i64,
        valid_counts,
        batch,
        seq,
        dim,
        strategy,
    } = prep;

    let ids_tensor = ort::value::Tensor::from_array(([batch, seq], input_ids))?;
    let mask_tensor = ort::value::Tensor::from_array(([batch, seq], attention_mask))?;

    let mut inputs = ort::inputs![
        "input_ids" => ids_tensor,
        "attention_mask" => mask_tensor,
    ];

    if let Some(tt) = token_type_ids_i64 {
        let tt_tensor = ort::value::Tensor::from_array(([batch, seq], tt))?;
        inputs.extend(ort::inputs!["token_type_ids" => tt_tensor]);
    }

    let run_options = RunOptions::new()?;
    let outputs = session.run_async(inputs, &run_options)?.await?;
    extract_pooled(&outputs, batch, seq, dim, &valid_counts, &strategy)
}

fn extract_pooled(
    outputs: &SessionOutputs<'_>,
    batch: usize,
    seq: usize,
    dim: usize,
    valid_counts: &[usize],
    strategy: &PoolingStrategy,
) -> Result<Vec<Vec<f32>>> {
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

    let stride = seq * dim;
    let mut results = Vec::with_capacity(batch);
    for (i, &count) in valid_counts.iter().enumerate().take(batch) {
        let row_start = i * stride;
        let row_data = &data[row_start..row_start + stride];
        let mut pooled = pool_row(strategy, row_data, dim, count);
        normalize_l2(&mut pooled);
        results.push(pooled);
    }
    Ok(results)
}
