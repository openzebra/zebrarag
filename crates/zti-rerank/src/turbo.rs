use std::cmp::Ordering;

use anyhow::Result;
use bitpolar::turbo::{TurboCode, TurboQuantizer};

pub struct RerankParams {
    pub bits: u8,
    pub projections: usize,
    pub seed: u64,
}

impl Default for RerankParams {
    fn default() -> Self {
        // bits=3 matches the bitpolar paper's "provably unbiased inner
        // product" setting (ICLR 2026). projections=0 → with_params picks
        // (2 * dim).max(32), the paper's lower bound.
        Self {
            bits: 3,
            projections: 0,
            seed: 42,
        }
    }
}

pub struct TurboReranker {
    quantizer: TurboQuantizer,
}

impl TurboReranker {
    pub fn new(dim: usize) -> Result<Self> {
        Self::with_params(dim, RerankParams::default())
    }

    pub fn with_params(dim: usize, params: RerankParams) -> Result<Self> {
        let projections = if params.projections == 0 {
            (2 * dim).max(32)
        } else {
            params.projections
        };
        let quantizer = TurboQuantizer::new(dim, params.bits, projections, params.seed)
            .map_err(|e| anyhow::anyhow!("TurboQuantizer init: {}", e))?;
        Ok(Self { quantizer })
    }

    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let code = self
            .quantizer
            .encode(vector)
            .map_err(|e| anyhow::anyhow!("TurboQuant encode: {}", e))?;
        Ok(code.to_compact_bytes())
    }

    pub fn score(&self, code_bytes: &[u8], query: &[f32]) -> Option<f32> {
        let code = TurboCode::from_compact_bytes(code_bytes).ok()?;
        self.quantizer.inner_product_estimate(&code, query).ok()
    }

    pub fn rerank(&self, candidates: &[(&[u8], f32)], query: &[f32]) -> Vec<(usize, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut scores: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, (code_bytes, initial_score))| {
                let code = TurboCode::from_compact_bytes(code_bytes).ok()?;
                let ip = self.quantizer.inner_product_estimate(&code, query).ok()?;
                let combined = initial_score + ip;
                Some((i, combined))
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores
    }

    pub fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    pub fn quantizer(&self) -> &TurboQuantizer {
        &self.quantizer
    }
}
