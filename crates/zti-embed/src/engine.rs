use anyhow::Result;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::model_registry::{ModelProfile, download_model, resolve_profile};
use crate::normalize::normalize_l2;
use crate::pooling::{PoolingStrategy, pool};
use crate::tokenizer::{Tokenizer, Tokenized};
use zti_hw::{Hardware, probe, register};

pub struct EmbedEngine {
    session: std::sync::Mutex<Session>,
    tokenizer: Tokenizer,
    profile: ModelProfile,
    hardware: Hardware,
}

impl EmbedEngine {
    pub fn load(model_id: &str) -> Result<Self> {
        let hw = probe();
        tracing::info!(device = ?hw.device, cpus = hw.cpus, "probing hardware");
        Self::load_with_device(model_id, &hw)
    }

    pub fn load_with_device(model_id: &str, hw: &Hardware) -> Result<Self> {
        download_model(model_id)?;
        let profile = resolve_profile(model_id)?;

        tracing::info!(path = %profile.onnx_path.display(), "loading ONNX model");

        let eps = register();
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {}", e))?
            .with_intra_threads(hw.cpus as usize)
            .map_err(|e| anyhow::anyhow!("intra threads: {}", e))?
            .with_execution_providers(eps)
            .map_err(|e| anyhow::anyhow!("execution providers: {}", e))?;

        let session = builder.commit_from_file(&profile.onnx_path)?;
        let tokenizer = Tokenizer::from_file(&profile.tokenizer_path)?;

        tracing::info!(dim = profile.dim, model = %model_id, "embed engine ready");

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            profile,
            hardware: hw.clone(),
        })
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let max_len = self.profile.max_length;
        let tokenized: Vec<Tokenized> = texts
            .iter()
            .map(|t| self.tokenizer.encode(t))
            .collect::<Result<Vec<_>>>()?;

        let max_seq = tokenized
            .iter()
            .map(|t| t.ids.len().min(max_len))
            .max()
            .unwrap_or(max_len);

        let batch = texts.len();
        let seq = max_seq;
        let dim = self.profile.dim;

        let mut input_ids = vec![0i64; batch * seq];
        let mut attention_mask = vec![0i64; batch * seq];

        for (i, tok) in tokenized.iter().enumerate() {
            let len = tok.ids.len().min(seq);
            for j in 0..len {
                input_ids[i * seq + j] = tok.ids[j] as i64;
                attention_mask[i * seq + j] = tok.mask[j] as i64;
            }
        }

        let ids_tensor = ort::value::Tensor::from_array(([batch, seq], input_ids))?;
        let mask_tensor = ort::value::Tensor::from_array(([batch, seq], attention_mask.clone()))?;

        let inputs = ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        ];

        let last_hidden = {
            let mut session = self.session.lock().unwrap();
            let outputs = session.run(inputs)?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            data.to_vec()
        };

        let strategy = PoolingStrategy::from(self.profile.pooling);

        let mut results = Vec::with_capacity(batch);
        for i in 0..batch {
            let row: Vec<&[f32]> = (0..seq)
                .map(|j| {
                    let offset = (i * seq + j) * dim;
                    &last_hidden[offset..offset + dim]
                })
                .collect();
            let mask_row: Vec<u32> = (0..seq)
                .map(|j| attention_mask[i * seq + j] as u32)
                .collect();

            let mut pooled = pool(&strategy, &row, &mask_row)?;
            normalize_l2(&mut pooled);
            results.push(pooled);
        }

        Ok(results)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = match &self.profile.query_prefix {
            Some(prefix) => format!("{prefix}{text}"),
            None => text.to_string(),
        };
        let batch = self.embed_batch(&[&input])?;
        batch.into_iter().next().ok_or_else(|| anyhow::anyhow!("no embedding produced"))
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
