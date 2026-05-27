use std::f32::consts::{FRAC_PI_2, TAU};

use anyhow::Result;
use bitpolar::{StoredRotation, TurboQuantizer};
use candle_core::{DType, Device, Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::turbo::TurboReranker;

pub const BATCH_SIZE: usize = 4096;
const COMPACT_VERSION: u8 = 0x01;

#[derive(Default)]
pub struct TurboCodeBatch {
    pub chunk_ids: Vec<[u8; 16]>,
    pub radii: Vec<f32>,
    pub angle_indices: Vec<u16>,
    pub norms: Vec<f32>,
    pub signs: Vec<u8>,
}

impl TurboCodeBatch {
    pub fn with_capacity(n: usize, dim_over_2: usize, sign_bytes_per_code: usize) -> Self {
        Self {
            chunk_ids: Vec::with_capacity(n),
            radii: Vec::with_capacity(n * dim_over_2),
            angle_indices: Vec::with_capacity(n * dim_over_2),
            norms: Vec::with_capacity(n),
            signs: Vec::with_capacity(n * sign_bytes_per_code),
        }
    }

    pub fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunk_ids.is_empty()
    }

    pub fn clear(&mut self) {
        self.chunk_ids.clear();
        self.radii.clear();
        self.angle_indices.clear();
        self.norms.clear();
        self.signs.clear();
    }
}

pub struct GpuTurboScorer {
    device: Device,
    dim_over_2: usize,
    num_projections: usize,
    scale_factor: f32,
    sign_bytes_per_code: usize,
    cos_table: Tensor,
    sin_table: Tensor,
    qjl_proj: Tensor,
    rotation: StoredRotation,
    pre_signs_flat: Vec<f32>,
}

impl GpuTurboScorer {
    pub fn from_reranker(reranker: &TurboReranker, device: &Device) -> Result<Self> {
        let q: &TurboQuantizer = reranker.quantizer();
        let dim = q.dim();
        let bits = q.bits();
        let num_projections = q.projections();
        let dim_over_2 = dim / 2;
        let sign_bytes_per_code = num_projections.div_ceil(8);
        let scale_factor = FRAC_PI_2.sqrt() / num_projections as f32;

        let two_bits = 1u32 << bits;
        let mut cos_host = Vec::with_capacity(two_bits as usize);
        let mut sin_host = Vec::with_capacity(two_bits as usize);
        for i in 0..two_bits {
            let angle = (i as f32) * TAU / two_bits as f32;
            cos_host.push(angle.cos());
            sin_host.push(angle.sin());
        }
        let cos_table = Tensor::from_slice(&cos_host, two_bits as usize, device)?;
        let sin_table = Tensor::from_slice(&sin_host, two_bits as usize, device)?;

        let qjl_host = build_qjl_matrix(dim, num_projections, q.seed().wrapping_add(1));
        let qjl_proj =
            Tensor::from_slice(&qjl_host, (num_projections, dim), device)?.to_dtype(DType::F32)?;

        let rotation = StoredRotation::new(dim, q.seed())?;

        Ok(Self {
            device: device.clone(),
            dim_over_2,
            num_projections,
            scale_factor,
            sign_bytes_per_code,
            cos_table,
            sin_table,
            qjl_proj,
            rotation,
            pre_signs_flat: Vec::with_capacity(BATCH_SIZE * num_projections),
        })
    }

    pub fn pre_rotate(&self, query: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim_over_2 * 2);
        self.rotation.apply_slice(query, &mut out);
        out
    }

    pub fn dim_over_2(&self) -> usize {
        self.dim_over_2
    }

    pub fn sign_bytes_per_code(&self) -> usize {
        self.sign_bytes_per_code
    }

    pub fn score_batch(
        &mut self,
        batch: &TurboCodeBatch,
        rotated_query: &[f32],
    ) -> Result<Vec<([u8; 16], f32)>> {
        let n = batch.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        tracing::trace!(
            n = n,
            device = ?self.device,
            dim = self.dim_over_2 * 2,
            proj = self.num_projections,
            "score_batch: GPU QJL matmul",
        );

        let dim = self.dim_over_2 * 2;
        let rotated_query_t = Tensor::from_slice(rotated_query, dim, &self.device)?;

        // Step 1: QJL projection of query — keep result on GPU.
        let projected = self
            .qjl_proj
            .matmul(&rotated_query_t.unsqueeze(1)?)?
            .squeeze(1)?;

        // Step 2: polar decode on GPU (unchanged from original).
        let radii_t = Tensor::from_slice(&batch.radii, (n, self.dim_over_2), &self.device)?
            .to_dtype(DType::F32)?;
        let angle_i64: Vec<i64> = batch.angle_indices.iter().map(|&v| v as i64).collect();
        let angle_t = Tensor::from_slice(&angle_i64, n * self.dim_over_2, &self.device)?;

        let cos_vals = self
            .cos_table
            .index_select(&angle_t, 0)?
            .reshape((n, self.dim_over_2))?;
        let sin_vals = self
            .sin_table
            .index_select(&angle_t, 0)?
            .reshape((n, self.dim_over_2))?;

        let x_comps = (radii_t.clone() * cos_vals)?;
        let y_comps = (radii_t * sin_vals)?;

        let decoded =
            Tensor::cat(&[&x_comps.unsqueeze(2)?, &y_comps.unsqueeze(2)?], 2)?.reshape((n, dim))?;

        let polar_ip = decoded
            .broadcast_mul(&rotated_query_t)?
            .sum(1)?
            .to_vec1::<f32>()?;

        // Step 3: QJL sum — branchless bit unpack on CPU, then GPU matmul.
        let proj_count = self.num_projections;
        let sign_stride = self.sign_bytes_per_code;

        self.pre_signs_flat.clear();
        for i in 0..n {
            let base = i * sign_stride;
            for p in 0..proj_count {
                let byte_idx = p >> 3;
                let bit_idx = p & 7;
                let bit = (batch.signs[base + byte_idx] >> bit_idx) & 1;
                self.pre_signs_flat.push((bit as f32) * 2.0 - 1.0);
            }
        }

        let signs_t = Tensor::from_slice(
            &self.pre_signs_flat,
            (n, proj_count),
            &self.device,
        )?;
        // projected is [proj_count]; reshape to [proj_count, 1] for matmul
        let qjl_sums_t = signs_t.matmul(&projected.unsqueeze(1)?)?.squeeze(1)?;
        let qjl_sums = qjl_sums_t.to_vec1::<f32>()?;

        // Step 4: combine polar + QJL scores.
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let qjl_ip = batch.norms[i] * self.scale_factor * qjl_sums[i];
            scores.push((batch.chunk_ids[i], polar_ip[i] + qjl_ip));
        }
        Ok(scores)
    }
}

fn build_qjl_matrix(dim: usize, projections: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..projections * dim)
        .map(|_| <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) as f32)
        .collect()
}

pub fn parse_turbo_code_into(
    bytes: &[u8],
    batch: &mut TurboCodeBatch,
    chunk_id: &[u8; 16],
) -> bool {
    if bytes.len() < 5 || bytes[0] != COMPACT_VERSION {
        return false;
    }
    let polar_len = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
    let polar_end = 5usize.saturating_add(polar_len);
    if bytes.len() < polar_end {
        return false;
    }
    let polar_bytes = &bytes[5..polar_end];
    if polar_bytes.len() < 4 || polar_bytes[0] != COMPACT_VERSION {
        return false;
    }
    let num_pairs = u16::from_le_bytes([polar_bytes[2], polar_bytes[3]]) as usize;
    let radii_end = 4 + num_pairs * 4;
    let angles_end = radii_end + num_pairs * 2;
    if polar_bytes.len() < angles_end {
        return false;
    }

    let qjl_bytes = &bytes[polar_end..];
    if qjl_bytes.len() < 8 || qjl_bytes[0] != COMPACT_VERSION {
        return false;
    }
    let norm = f32::from_le_bytes([qjl_bytes[3], qjl_bytes[4], qjl_bytes[5], qjl_bytes[6]]);

    batch.chunk_ids.push(*chunk_id);
    let radii = &polar_bytes[4..radii_end];
    batch.radii.reserve(num_pairs);
    for b in radii.chunks_exact(4) {
        batch
            .radii
            .push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    let angle_bytes = &polar_bytes[radii_end..angles_end];
    batch.angle_indices.reserve(num_pairs);
    for b in angle_bytes.chunks_exact(2) {
        batch.angle_indices.push(u16::from_le_bytes([b[0], b[1]]));
    }
    batch.norms.push(norm);
    batch.signs.extend_from_slice(&qjl_bytes[7..]);
    true
}
