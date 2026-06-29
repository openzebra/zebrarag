use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use bitpolar::{StoredRotation, TurboQuantizer};
use candle_core::{DType, Device, DeviceLocation, Tensor};
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

/// Immutable, device-resident state shared across queries via `Arc`.
/// Deterministic from (quantizer params, device); built once.
pub struct GpuTurboCore {
    device: Device,
    dim_over_2: usize,
    num_projections: usize,
    scale_factor: f32,
    sign_bytes_per_code: usize,
    cos_table: Tensor,
    sin_table: Tensor,
    qjl_proj: Tensor,
    rotation: StoredRotation,
}

impl GpuTurboCore {
    pub fn from_reranker(reranker: &TurboReranker, device: &Device) -> Result<Arc<Self>> {
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

        Ok(Arc::new(Self {
            device: device.clone(),
            dim_over_2,
            num_projections,
            scale_factor,
            sign_bytes_per_code,
            cos_table,
            sin_table,
            qjl_proj,
            rotation,
        }))
    }

    #[inline]
    pub const fn num_projections(&self) -> usize {
        self.num_projections
    }

    #[inline]
    pub const fn dim_over_2(&self) -> usize {
        self.dim_over_2
    }

    #[inline]
    pub const fn sign_bytes_per_code(&self) -> usize {
        self.sign_bytes_per_code
    }

    #[inline]
    pub fn pre_rotate_into(&self, query: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.dim_over_2 * 2);
        self.rotation.apply_slice(query, out);
    }
}

/// Per-query scratch — reused buffers, never shared between tasks.
#[derive(Default)]
pub struct GpuTurboScratch {
    pub pre_signs_flat: Vec<f32>,
    pub angle_i64: Vec<i64>,
    pub scores: Vec<([u8; 16], f32)>,
}

impl GpuTurboScratch {
    pub fn with_capacity(num_projections: usize, dim_over_2: usize) -> Self {
        Self {
            pre_signs_flat: Vec::with_capacity(BATCH_SIZE * num_projections),
            angle_i64: Vec::with_capacity(BATCH_SIZE * dim_over_2),
            scores: Vec::with_capacity(BATCH_SIZE),
        }
    }

    /// Shrink any overallocated buffers back to the ceiling implied by
    /// `core`'s dimensions. Only triggers when capacity exceeds 2× the
    /// ceiling — the common case (capacity already at ceiling) is a
    /// handful of integer comparisons with no reallocation.
    ///
    /// This prevents high-water-mark memory bloat in the long-running
    /// daemon: if an anomalous batch forces a `Vec` to grow, the extra
    /// capacity is released on the next call rather than being held
    /// indefinitely.
    pub fn bound_to_core(&mut self, core: &GpuTurboCore) {
        let ceiling = BATCH_SIZE * core.num_projections();
        if self.pre_signs_flat.capacity() > ceiling * 2 {
            self.pre_signs_flat.shrink_to(ceiling);
        }
        let ceiling = BATCH_SIZE * core.dim_over_2();
        if self.angle_i64.capacity() > ceiling * 2 {
            self.angle_i64.shrink_to(ceiling);
        }
        let ceiling = BATCH_SIZE;
        if self.scores.capacity() > ceiling * 2 {
            self.scores.shrink_to(ceiling);
        }
    }
}

#[derive(PartialEq, Eq)]
struct CoreKey {
    dim: usize,
    seed: u64,
    location: DeviceLocation,
}

/// Mirrors `AnnCache::get_or_build`. Keyed on (dim, seed, device-location) —
/// the three things `GpuTurboCore` actually depends on.
#[derive(Default)]
pub struct TurboScorerCache {
    inner: Mutex<Option<(CoreKey, Arc<GpuTurboCore>)>>,
}

impl TurboScorerCache {
    pub fn get_or_build(
        &self,
        reranker: &TurboReranker,
        device: &Device,
    ) -> Result<Arc<GpuTurboCore>> {
        let key = CoreKey {
            dim: reranker.dim(),
            seed: reranker.quantizer().seed(),
            location: device.location(),
        };
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((cached_key, core)) = guard.as_ref()
            && *cached_key == key
        {
            return Ok(Arc::clone(core));
        }
        let core = GpuTurboCore::from_reranker(reranker, device)?;
        *guard = Some((key, Arc::clone(&core)));
        Ok(core)
    }
}

/// 256-entry lookup: byte → 8 sign floats ({-1.0, +1.0}).
/// Indexed by the raw sign byte from a turbo code; turns the 6 M-iteration
/// per-batch bit-unpack loop into a table read + `extend_from_slice`.
const SIGN_LOOKUP: [[f32; 8]; 256] = {
    let mut table = [[0.0f32; 8]; 256];
    let mut b = 0u8;
    loop {
        let mut bit = 0;
        while bit < 8 {
            table[b as usize][bit] = if (b >> bit) & 1 != 0 { 1.0 } else { -1.0 };
            bit += 1;
        }
        if b == 255 {
            break;
        }
        b = b.wrapping_add(1);
    }
    table
};

/// Zero-alloc batch scoring using pre-built core and reusable scratch.
/// Returns a slice into `scratch.scores` — values must be consumed before
/// the next call to `score_batch` with the same scratch.
pub fn score_batch<'s>(
    core: &GpuTurboCore,
    scratch: &'s mut GpuTurboScratch,
    batch: &TurboCodeBatch,
    rotated_query: &[f32],
) -> Result<&'s [([u8; 16], f32)]> {
    let n = batch.len();
    if n == 0 {
        scratch.scores.clear();
        return Ok(&scratch.scores);
    }

    tracing::trace!(
        n = n,
        device = ?core.device,
        dim = core.dim_over_2 * 2,
        proj = core.num_projections,
        "score_batch: GPU QJL matmul",
    );

    let dim = core.dim_over_2 * 2;
    let rotated_query_t = Tensor::from_slice(rotated_query, dim, &core.device)?;

    // Step 1: QJL projection of query — keep result on GPU.
    let projected = core
        .qjl_proj
        .matmul(&rotated_query_t.unsqueeze(1)?)?
        .squeeze(1)?;

    // Step 2: polar decode on GPU.
    let radii_t = Tensor::from_slice(&batch.radii, (n, core.dim_over_2), &core.device)?
        .to_dtype(DType::F32)?;

    scratch.angle_i64.clear();
    scratch
        .angle_i64
        .extend(batch.angle_indices.iter().map(|&v| i64::from(v)));
    let angle_t = Tensor::from_slice(&scratch.angle_i64, n * core.dim_over_2, &core.device)?;

    let cos_vals = core
        .cos_table
        .index_select(&angle_t, 0)?
        .reshape((n, core.dim_over_2))?;
    let sin_vals = core
        .sin_table
        .index_select(&angle_t, 0)?
        .reshape((n, core.dim_over_2))?;

    let x_comps = (&radii_t * &cos_vals)?;
    let y_comps = (&radii_t * &sin_vals)?;

    let decoded =
        Tensor::cat(&[&x_comps.unsqueeze(2)?, &y_comps.unsqueeze(2)?], 2)?.reshape((n, dim))?;

    // Keep on GPU — single readback at the end.
    let polar_ip_t = decoded.broadcast_mul(&rotated_query_t)?.sum(1)?;

    // Step 3: QJL sign expansion — lookup-table driven, no bit-unpack loop.
    let proj_count = core.num_projections;
    let sign_stride = core.sign_bytes_per_code;
    let full_bytes = proj_count >> 3;
    let rem_bits = proj_count & 7;

    scratch.pre_signs_flat.clear();
    for i in 0..n {
        let base = i * sign_stride;
        let bytes = batch
            .signs
            .get(base..base.saturating_add(sign_stride))
            .unwrap_or(&[]);
        if bytes.len() < sign_stride {
            scratch
                .pre_signs_flat
                .resize(scratch.pre_signs_flat.len() + proj_count, 0.0);
            continue;
        }
        for b in 0..full_bytes {
            scratch
                .pre_signs_flat
                .extend_from_slice(&SIGN_LOOKUP[bytes[b] as usize]);
        }
        if rem_bits > 0 {
            scratch
                .pre_signs_flat
                .extend_from_slice(&SIGN_LOOKUP[bytes[full_bytes] as usize][..rem_bits]);
        }
    }

    let signs_t = Tensor::from_slice(&scratch.pre_signs_flat, (n, proj_count), &core.device)?;
    let qjl_sums_t = signs_t.matmul(&projected.unsqueeze(1)?)?.squeeze(1)?;

    // Step 4: combine polar + QJL on-device → single GPU→CPU readback.
    let norms_t = Tensor::from_slice(&batch.norms, n, &core.device)?;
    let scale_t = Tensor::new(&[core.scale_factor], &core.device)?;
    let norms_scaled = norms_t.broadcast_mul(&scale_t)?;
    let combined = (&polar_ip_t + (&norms_scaled * &qjl_sums_t)?)?;
    let combined_vec = combined.to_vec1::<f32>()?;

    scratch.scores.clear();
    scratch.scores.reserve(n);
    scratch.scores.extend(
        batch
            .chunk_ids
            .iter()
            .zip(&combined_vec)
            .map(|(&id, &score)| (id, score)),
    );

    // Release any high-water-mark overcapacity (long-running daemon guard).
    scratch.bound_to_core(core);

    Ok(&scratch.scores)
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

#[cfg(test)]
mod tq_tests {
    use super::*;
    use std::f32;

    fn test_dim() -> usize {
        128
    }

    fn test_chunk_id(val: u8) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[0] = val;
        id
    }

    fn make_reranker(dim: usize) -> crate::turbo::TurboReranker {
        crate::turbo::TurboReranker::new(dim).expect("TurboReranker::new should succeed")
    }

    fn unit_vector(dim: usize) -> Vec<f32> {
        let scale = (dim as f32).sqrt().recip();
        vec![scale; dim]
    }

    #[test]
    fn build_qjl_matrix_size() {
        let dim = 16;
        let proj = 32;
        let m = build_qjl_matrix(dim, proj, 42);
        assert_eq!(m.len(), proj * dim);
    }

    #[test]
    fn build_qjl_matrix_deterministic() {
        let m1 = build_qjl_matrix(16, 32, 42);
        let m2 = build_qjl_matrix(16, 32, 42);
        assert_eq!(m1, m2);
    }

    #[test]
    fn build_qjl_matrix_different_seed() {
        let m1 = build_qjl_matrix(16, 32, 42);
        let m2 = build_qjl_matrix(16, 32, 99);
        assert_ne!(m1, m2);
    }

    #[test]
    fn turbo_code_batch_capacities() {
        let n = 10;
        let dim_over_2 = 64;
        let sign_bytes = 8;
        let batch = TurboCodeBatch::with_capacity(n, dim_over_2, sign_bytes);
        assert_eq!(batch.chunk_ids.capacity(), n);
        assert!(batch.radii.capacity() >= n * dim_over_2);
        assert!(batch.angle_indices.capacity() >= n * dim_over_2);
        assert!(batch.norms.capacity() >= n);
        assert!(batch.signs.capacity() >= n * sign_bytes);
    }

    #[test]
    fn turbo_code_batch_clear() {
        let mut batch = TurboCodeBatch {
            chunk_ids: vec![test_chunk_id(1)],
            radii: vec![0.5f32; 4],
            angle_indices: vec![0u16; 4],
            norms: vec![1.0f32],
            signs: vec![0xabu8; 8],
        };
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
        batch.clear();
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());
        assert!(batch.radii.is_empty());
        assert!(batch.angle_indices.is_empty());
        assert!(batch.norms.is_empty());
        assert!(batch.signs.is_empty());
    }

    #[test]
    fn parse_too_short_false() {
        let mut batch = TurboCodeBatch::default();
        let id = test_chunk_id(1);
        assert!(!parse_turbo_code_into(&[0x01, 0x00], &mut batch, &id));
        assert!(batch.is_empty());
    }

    #[test]
    fn parse_wrong_version_false() {
        let mut batch = TurboCodeBatch::default();
        let id = test_chunk_id(1);
        let data = [0x02u8; 16];
        assert!(!parse_turbo_code_into(&data, &mut batch, &id));
        assert!(batch.is_empty());
    }

    #[test]
    fn parse_truncated_polar_false() {
        let mut batch = TurboCodeBatch::default();
        let id = test_chunk_id(1);
        let polar_len = 100u32;
        let mut header = Vec::with_capacity(5);
        header.push(COMPACT_VERSION);
        header.extend_from_slice(&polar_len.to_le_bytes());
        header.extend_from_slice(&[0u8; 10]);
        assert!(!parse_turbo_code_into(&header, &mut batch, &id));
        assert!(batch.is_empty());
    }

    #[test]
    fn parse_round_trip_single() -> Result<()> {
        let r = make_reranker(test_dim());
        let v = unit_vector(test_dim());
        let code_bytes = r.encode(&v)?;
        let id = test_chunk_id(42);
        let mut batch = TurboCodeBatch::default();
        let ok = parse_turbo_code_into(&code_bytes, &mut batch, &id);
        assert!(ok, "should parse valid code");
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.chunk_ids[0], id);
        assert_eq!(batch.norms.len(), 1);
        assert!(!batch.radii.is_empty());
        assert!(!batch.angle_indices.is_empty());
        assert!(!batch.signs.is_empty());
        Ok(())
    }

    #[test]
    fn parse_round_trip_multiple() -> Result<()> {
        let r = make_reranker(test_dim());
        let v = unit_vector(test_dim());
        let code1 = r.encode(&v)?;
        let code2 = r.encode(&v)?;
        let id1 = test_chunk_id(1);
        let id2 = test_chunk_id(2);
        let mut batch = TurboCodeBatch::default();
        assert!(parse_turbo_code_into(&code1, &mut batch, &id1));
        assert!(parse_turbo_code_into(&code2, &mut batch, &id2));
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.chunk_ids[0], id1);
        assert_eq!(batch.chunk_ids[1], id2);
        assert_eq!(batch.norms.len(), 2);
        Ok(())
    }
}
