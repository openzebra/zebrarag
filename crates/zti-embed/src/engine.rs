use std::sync::{Arc, Mutex};

use anyhow::Result;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::ValueType;

use crate::model_registry::{ModelProfile, download_model, resolve_profile};
use crate::normalize::normalize_l2;
use crate::pooling::{PoolingStrategy, pool_row};
use crate::tokenizer::Tokenizer;
use zti_hw::{Hardware, probe, register};

pub struct EmbedEngine {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    profile: ModelProfile,
    hardware: Hardware,
}

struct Prepared {
    input_ids: Vec<i64>,
    attention_mask_i64: Vec<i64>,
    attention_mask_u32: Vec<u32>,
    batch: usize,
    seq: usize,
    dim: usize,
    strategy: PoolingStrategy,
}

impl EmbedEngine {
    pub fn load(model_id: &str) -> Result<Self> {
        let hw = probe();
        tracing::info!(device = ?hw.device, cpus = hw.cpus, "probing hardware");
        Self::load_with_device(model_id, &hw)
    }

    pub fn load_with_device(model_id: &str, hw: &Hardware) -> Result<Self> {
        let _ = download_model(model_id)?;
        let mut profile = resolve_profile(model_id)?;

        tracing::info!(path = %profile.onnx_path.display(), "loading ONNX model");

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {}", e))?
            .with_intra_threads(hw.cpus)
            .map_err(|e| anyhow::anyhow!("intra threads: {}", e))?
            .with_execution_providers(register())
            .map_err(|e| anyhow::anyhow!("execution providers: {}", e))?;

        let session = builder.commit_from_file(&profile.onnx_path)?;
        let tokenizer = Tokenizer::from_file(&profile.tokenizer_path)?;

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
        if let Some(max_len) = tokenizer.truncation_max_length() {
            profile.max_length = max_len;
        }
        if profile.dim == 0 {
            // Shape was dynamic. Probe by running a single token through the
            // session and reading the resulting last-axis length.
            let engine_tmp = Self {
                session: Arc::new(Mutex::new(session)),
                tokenizer,
                profile: profile.clone(),
                hardware: *hw,
            };
            let probe = engine_tmp.embed_batch(&["a"])?;
            let probed = probe.first().map(|v| v.len()).unwrap_or(0);
            if probed == 0 {
                anyhow::bail!("could not determine embedding dimension for {}", model_id);
            }
            let mut profile = engine_tmp.profile;
            profile.dim = probed;
            tracing::info!(
                dim = probe.first().map(|v| v.len()).unwrap_or(0),
                "probed embedding dim by warmup"
            );
            return Ok(Self {
                session: engine_tmp.session,
                tokenizer: engine_tmp.tokenizer,
                profile,
                hardware: *hw,
            });
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
        })
    }

    fn prepare(&self, texts: &[&str]) -> Result<Option<Prepared>> {
        if texts.is_empty() {
            return Ok(None);
        }

        let max_len = self.profile.max_length;
        let batch = texts.len();
        let dim = self.profile.dim;

        let mut encs = Vec::with_capacity(batch);
        let mut max_seq = 0usize;
        for t in texts {
            let enc = self.tokenizer.encode(t)?;
            let len = enc.ids.len().min(max_len);
            if len > max_seq {
                max_seq = len;
            }
            encs.push(enc);
        }
        if max_seq == 0 {
            max_seq = 1;
        }
        let seq = max_seq;

        let total = batch * seq;
        let mut input_ids = vec![0i64; total];
        let mut attention_mask_i64 = vec![0i64; total];
        let mut attention_mask_u32 = vec![0u32; total];

        for (i, tok) in encs.iter().enumerate() {
            let len = tok.ids.len().min(seq);
            let base = i * seq;
            for j in 0..len {
                input_ids[base + j] = tok.ids[j] as i64;
                attention_mask_i64[base + j] = tok.mask[j] as i64;
                attention_mask_u32[base + j] = tok.mask[j];
            }
        }

        Ok(Some(Prepared {
            input_ids,
            attention_mask_i64,
            attention_mask_u32,
            batch,
            seq,
            dim,
            strategy: PoolingStrategy::from(self.profile.pooling),
        }))
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prep = match self.prepare(texts)? {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };
        run_and_pool(&self.session, prep)
    }

    pub async fn embed_batch_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prep = match self.prepare(texts)? {
            Some(p) => p,
            None => return Ok(Vec::new()),
        };
        let session = Arc::clone(&self.session);
        tokio::task::spawn_blocking(move || run_and_pool(&session, prep)).await?
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
}

fn run_and_pool(session: &Mutex<Session>, prep: Prepared) -> Result<Vec<Vec<f32>>> {
    let Prepared {
        input_ids,
        attention_mask_i64,
        attention_mask_u32,
        batch,
        seq,
        dim,
        strategy,
    } = prep;

    let ids_tensor = ort::value::Tensor::from_array(([batch, seq], input_ids))?;
    let mask_tensor = ort::value::Tensor::from_array(([batch, seq], attention_mask_i64))?;

    let inputs = ort::inputs![
        "input_ids" => ids_tensor,
        "attention_mask" => mask_tensor,
    ];

    let mut session = session
        .lock()
        .map_err(|_| anyhow::anyhow!("embed session mutex poisoned"))?;
    let outputs = session.run(inputs)?;
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

    let stride = seq * dim;
    let mut results = Vec::with_capacity(batch);
    for i in 0..batch {
        let row_start = i * stride;
        let row_data = &data[row_start..row_start + stride];
        let row_mask = &attention_mask_u32[i * seq..(i + 1) * seq];
        let mut pooled = pool_row(&strategy, row_data, dim, seq, row_mask);
        normalize_l2(&mut pooled);
        results.push(pooled);
    }
    Ok(results)
}
