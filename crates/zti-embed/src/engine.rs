use std::sync::Arc;
use std::borrow::Cow;
use std::sync::Mutex;

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

use crate::batch::{BATCH_BUCKETS, BATCH_CEILING, SEQ_BUCKETS, next_bucket};
use crate::model_registry::{ModelProfile, resolve_profile};
use crate::normalize::normalize_l2;
use crate::pooling::{PoolingStrategy, pool_row_into};
use crate::tokenizer::{Tokenized, Tokenizer};
use zti_hw::{Hardware, probe, candle_device};

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

#[derive(Debug, Clone, Default)]
pub struct LoadOverrides<'a> {
    pub query_prefix: Option<&'a str>,
    pub passage_prefix: Option<&'a str>,
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

struct State {
    model: BertModel,
    device: candle_core::Device,
    scratch: Scratch,
}

pub struct EmbedEngine {
    state: Arc<Mutex<State>>,
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
        let mut profile = resolve_profile(
            model_id,
            opts.query_prefix,
            opts.passage_prefix,
        )?;

        tracing::info!(path = %profile.weights_path.display(), "loading safetensors model");

        let device = candle_device(&hw);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[&profile.weights_path],
                DType::F32,
                &device,
            )?
        };

        let config = BertConfig {
            hidden_size: profile.hidden_size,
            num_hidden_layers: profile.num_hidden_layers,
            num_attention_heads: profile.num_attention_heads,
            intermediate_size: profile.intermediate_size,
            max_position_embeddings: profile.max_length,
            ..Default::default()
        };

        let model = BertModel::load(vb, &config)?;
        let tokenizer = Tokenizer::from_file(&profile.tokenizer_path)?;

        if let Some(tok_limit) = tokenizer.truncation_max_length() {
            profile.max_length = profile.max_length.min(tok_limit);
        }

        if profile.dim == 0 {
            let enc = tokenizer.encode("a")?;
            if enc.ids.is_empty() {
                anyhow::bail!("warmup produced no tokens");
            }
            let ids = Tensor::from_slice(&enc.ids, (1, enc.ids.len()), &device)?;
            let mask = Tensor::ones((1, enc.ids.len()), DType::F32, &device)?;
            let out = model.forward(&ids, &mask, None)?;
            profile.dim = out.dims()[2];
            tracing::info!(dim = profile.dim, "probed embedding dim");
        }

        tracing::info!(
            dim = profile.dim,
            max_len = profile.max_length,
            device = ?hw.device,
            model = %model_id,
            "embed engine ready"
        );

        let scratch = Scratch::with_capacity(BATCH_CEILING, profile.max_length, profile.dim);

        Ok(Self {
            state: Arc::new(Mutex::new(State { model, device, scratch })),
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

    pub fn recommended_batch_size(&self) -> usize {
        crate::batch::recommended_batch_size(&self.profile, &self.hardware)
    }

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

        let mut state = self.state.lock().expect("embed state poisoned");
        let State { model, device, scratch } = &mut *state;
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
                dim: self.profile.dim,
            },
            &PoolingStrategy::from(self.profile.pooling),
        )?;

        Ok(Pooled {
            data: scratch.pooled_flat.clone(),
            dim: self.profile.dim,
            batch: real_batch,
        })
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
    model: &BertModel,
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

    let ids = Tensor::from_slice(
        &scratch.input_ids[..batch * seq],
        (batch, seq),
        device,
    )?;
    let mask = Tensor::from_slice(
        &scratch.attention_mask[..batch * seq],
        (batch, seq),
        device,
    )?
    .to_dtype(DType::F32)?;

    let output = model.forward(&ids, &mask, None)?;
    let output = output.to_device(&candle_core::Device::Cpu)?.to_dtype(DType::F32)?;
    let data = output.to_vec3::<f32>()?;

    scratch.pooled_flat.clear();
    scratch.pooled_flat.resize(real_batch * dim, 0.0);

    for (i, row) in data.iter().enumerate().take(real_batch) {
        let out = &mut scratch.pooled_flat[i * dim..(i + 1) * dim];
        pool_row_into(strategy, &row.concat(), scratch.valid_counts[i], out);
        normalize_l2(out);
    }

    Ok(())
}
