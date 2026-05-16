use std::cmp::Ordering;

use anyhow::Result;
use bitpolar::turbo::{TurboCode, TurboQuantizer};

pub struct TurboReranker {
    quantizer: TurboQuantizer,
    codes: Vec<TurboCode>,
    dim: usize,
}

impl TurboReranker {
    pub fn new(dim: usize) -> Result<Self> {
        let bits = 4;
        let projections = dim.max(4);
        let seed = 42u64;
        let quantizer = TurboQuantizer::new(dim, bits, projections, seed)
            .map_err(|e| anyhow::anyhow!("TurboQuantizer init: {}", e))?;
        Ok(Self {
            quantizer,
            codes: Vec::new(),
            dim,
        })
    }

    pub fn encode(&mut self, vector: &[f32]) -> Result<()> {
        let code = self
            .quantizer
            .encode(vector)
            .map_err(|e| anyhow::anyhow!("TurboQuant encode: {}", e))?;
        self.codes.push(code);
        Ok(())
    }

    pub fn push_code(&mut self, code: TurboCode) {
        self.codes.push(code);
    }

    pub fn rerank(&self, candidates: &[usize], query: &[f32]) -> Vec<(usize, f32)> {
        if self.codes.is_empty() || candidates.is_empty() {
            return Vec::new();
        }
        let mut scores: Vec<(usize, f32)> = candidates
            .iter()
            .filter_map(|&i| {
                let code = self.codes.get(i)?;
                self.quantizer
                    .inner_product_estimate(code, query)
                    .ok()
                    .map(|s| (i, s))
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}
