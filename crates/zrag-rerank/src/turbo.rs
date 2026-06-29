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
            .map(|(i, (code_bytes, initial_score))| {
                // A missing or undecodable turbo code must NOT drop the
                // candidate — it was legitimately surfaced by vector/lexical
                // search. Fall back to the initial score with no reranker
                // boost so recall is preserved; the candidate simply competes
                // on its fusion score alone.
                let boost = TurboCode::from_compact_bytes(code_bytes)
                    .ok()
                    .and_then(|code| self.quantizer.inner_product_estimate(&code, query).ok())
                    .unwrap_or(0.0);
                (i, initial_score + boost)
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

#[cfg(test)]
mod tq_tests {
    use super::*;

    fn test_dim() -> usize {
        128
    }

    fn make_reranker(dim: usize) -> TurboReranker {
        TurboReranker::new(dim).expect("TurboReranker::new should succeed")
    }

    fn unit_vector(dim: usize) -> Vec<f32> {
        let scale = (dim as f32).sqrt().recip();
        vec![scale; dim]
    }

    #[test]
    fn rerank_params_default() {
        let p = RerankParams::default();
        assert_eq!(p.bits, 3);
        assert_eq!(p.projections, 0);
        assert_eq!(p.seed, 42);
    }

    #[test]
    fn turbo_reranker_new_succeeds() -> Result<()> {
        let r = TurboReranker::new(test_dim())?;
        assert_eq!(r.dim(), test_dim());
        Ok(())
    }

    #[test]
    fn turbo_reranker_with_params() -> Result<()> {
        let p = RerankParams {
            bits: 4,
            projections: 64,
            seed: 99,
        };
        let r = TurboReranker::with_params(test_dim(), p)?;
        assert_eq!(r.dim(), test_dim());
        assert_eq!(r.quantizer().bits(), 4);
        assert_eq!(r.quantizer().seed(), 99);
        Ok(())
    }

    #[test]
    fn zero_projections_auto() -> Result<()> {
        let r = TurboReranker::new(test_dim())?;
        let expected = (2 * test_dim()).max(32);
        assert_eq!(r.quantizer().projections(), expected);
        Ok(())
    }

    #[test]
    fn zero_projections_low_dim() -> Result<()> {
        let r = TurboReranker::new(8)?;
        assert_eq!(r.quantizer().projections(), 32);
        Ok(())
    }

    #[test]
    fn dim_getter() -> Result<()> {
        for dim in [8, 16, 64, 128, 256] {
            let r = TurboReranker::new(dim)?;
            assert_eq!(r.dim(), dim);
        }
        Ok(())
    }

    #[test]
    fn encode_non_empty() -> Result<()> {
        let r = make_reranker(test_dim());
        let v = unit_vector(test_dim());
        let bytes = r.encode(&v)?;
        assert!(!bytes.is_empty());
        Ok(())
    }

    #[test]
    fn encode_deterministic() -> Result<()> {
        let p = RerankParams {
            bits: 3,
            projections: 64,
            seed: 42,
        };
        let r = TurboReranker::with_params(test_dim(), p)?;
        let v = unit_vector(test_dim());
        let a = r.encode(&v)?;
        let b = r.encode(&v)?;
        assert_eq!(a, b);
        Ok(())
    }

    #[test]
    fn score_self_high() -> Result<()> {
        let r = make_reranker(test_dim());
        let v = unit_vector(test_dim());
        let code_bytes = r.encode(&v)?;
        let score = r.score(&code_bytes, &v);
        assert!(score.is_some());
        let s = score.unwrap();
        assert!(s > 0.0, "self-score should be positive, got {s}");
        Ok(())
    }

    #[test]
    fn score_dissimilar_lower() -> Result<()> {
        let r = make_reranker(test_dim());
        let query = unit_vector(test_dim());
        let similar: Vec<f32> = query.clone();
        let mut dissimilar = query.clone();
        dissimilar[0] = -dissimilar[0];

        let code_similar = r.encode(&similar)?;
        let code_dissimilar = r.encode(&dissimilar)?;

        let score_sim = r.score(&code_similar, &query).unwrap_or(f32::MIN);
        let score_dis = r.score(&code_dissimilar, &query).unwrap_or(f32::MAX);
        assert!(
            score_sim > score_dis,
            "similar vector ({score_sim}) should score higher than dissimilar ({score_dis})"
        );
        Ok(())
    }

    #[test]
    fn score_garbage_none() {
        let r = make_reranker(test_dim());
        let garbage = [0xabu8; 32];
        let q = unit_vector(test_dim());
        assert!(r.score(&garbage, &q).is_none());
    }

    #[test]
    fn rerank_empty() {
        let r = make_reranker(test_dim());
        let q = unit_vector(test_dim());
        let result = r.rerank(&[], &q);
        assert!(result.is_empty());
    }

    #[test]
    fn rerank_orders_by_combined() -> Result<()> {
        let r = make_reranker(test_dim());
        let query = unit_vector(test_dim());
        let close = query.clone();
        let far: Vec<f32> = query.iter().map(|x| -x).collect();

        let code_close = r.encode(&close)?;
        let code_far = r.encode(&far)?;

        let candidates: Vec<(&[u8], f32)> = vec![(&code_far, 1.0), (&code_close, 0.0)];
        let ranked = r.rerank(&candidates, &query);
        assert_eq!(ranked.len(), 2);
        let close_idx = ranked.iter().position(|(i, _)| *i == 1).unwrap();
        let far_idx = ranked.iter().position(|(i, _)| *i == 0).unwrap();
        assert!(
            close_idx < far_idx,
            "close candidate (idx 1) should rank before far candidate (idx 0)"
        );
        Ok(())
    }

    #[test]
    fn rerank_keeps_invalid_unboosted() -> Result<()> {
        // Regression: a candidate whose turbo code is missing/garbage must
        // STAY searchable (fall back to its initial score, no boost) rather
        // than being silently dropped. Dropping shrank recall on documents
        // whose chunks lacked valid turbo codes.
        let r = make_reranker(test_dim());
        let query = unit_vector(test_dim());
        let v = unit_vector(test_dim());
        let code_valid = r.encode(&v)?;
        let garbage = [0xffu8; 8];

        let candidates: Vec<(&[u8], f32)> = vec![(&garbage, 1.0), (&code_valid, 0.0)];
        let ranked = r.rerank(&candidates, &query);
        assert_eq!(
            ranked.len(),
            2,
            "invalid code must stay searchable, just unboosted"
        );
        Ok(())
    }

    #[test]
    fn quantizer_accessor() -> Result<()> {
        let r = TurboReranker::new(64)?;
        assert_eq!(r.quantizer().dim(), 64);
        Ok(())
    }
}
