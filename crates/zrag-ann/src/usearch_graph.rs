use anyhow::{Result, anyhow};
use usearch::Index;
use usearch::IndexOptions;
use usearch::ffi::MetricKind;
use usearch::ffi::ScalarKind;

use crate::method::SearchParams;

pub type ChunkId = [u8; 16];

fn new_index(dim: usize, p: &SearchParams) -> Result<Index> {
    let opts = IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: p.m as usize,
        expansion_add: p.ef_construction as usize,
        expansion_search: p.ef_search as usize,
        multi: false,
    };
    Index::new(&opts).map_err(|e| anyhow!("usearch index creation: {e}"))
}

pub struct AnnIndex {
    inner: Index,
    chunk_ids: Vec<ChunkId>,
    dim: usize,
}

pub struct AnnIndexBuilder {
    inner: Index,
    chunk_ids: Vec<ChunkId>,
    dim: usize,
    idx: u64,
    err: Option<anyhow::Error>,
}

impl AnnIndexBuilder {
    pub fn new(dim: usize, p: &SearchParams) -> Result<Self> {
        let inner = new_index(dim, p)?;
        Ok(Self {
            inner,
            chunk_ids: Vec::new(),
            dim,
            idx: 0,
            err: None,
        })
    }

    pub fn add(&mut self, id: ChunkId, v: &[f32]) {
        if self.err.is_some() {
            return;
        }
        if let Err(e) = self.inner.add(self.idx, v) {
            self.err = Some(anyhow!("usearch add key {}: {e}", self.idx));
            return;
        }
        self.chunk_ids.push(id);
        self.idx += 1;
    }

    pub fn reserve(&mut self, capacity: usize) -> Result<()> {
        self.inner
            .reserve(capacity)
            .map_err(|e| anyhow!("usearch reserve: {e}"))?;
        if self.chunk_ids.capacity() < capacity {
            self.chunk_ids.reserve(capacity - self.chunk_ids.len());
        }
        Ok(())
    }

    pub fn build(self) -> Result<AnnIndex> {
        if let Some(e) = self.err {
            return Err(e);
        }
        Ok(AnnIndex {
            inner: self.inner,
            chunk_ids: self.chunk_ids,
            dim: self.dim,
        })
    }
}

impl AnnIndex {
    pub fn build(
        dim: usize,
        flat: &[f32],
        chunk_ids: Vec<ChunkId>,
        p: &SearchParams,
    ) -> Result<Self> {
        let n = chunk_ids.len();
        let inner = new_index(dim, p)?;
        inner
            .reserve(n.max(1_024))
            .map_err(|e| anyhow!("usearch reserve: {e}"))?;
        for i in 0..n {
            let v = &flat[i * dim..(i + 1) * dim];
            inner
                .add(i as u64, v)
                .map_err(|e| anyhow!("usearch add key {i}: {e}"))?;
        }
        Ok(Self {
            inner,
            chunk_ids,
            dim,
        })
    }

    #[inline]
    pub fn search(&self, query: &[f32], k: usize, out: &mut Vec<(ChunkId, f32)>) {
        out.clear();
        let n = self.chunk_ids.len();
        if n == 0 || k == 0 || query.len() != self.dim {
            return;
        }
        let matches = match self.inner.search(query, k.min(n)) {
            Ok(m) => m,
            Err(_) => return,
        };
        out.reserve(matches.keys.len());
        for (key, distance) in matches.keys.iter().zip(matches.distances.iter()) {
            let idx = *key as usize;
            if let Some(id) = self.chunk_ids.get(idx) {
                out.push((*id, 1.0 - distance));
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chunk_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_counting;

    fn test_params() -> SearchParams {
        SearchParams {
            method: crate::SearchMethod::Usearch,
            indexed_chunks: 0,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            num_partitions: 4,
            nprobes: 4,
            refine_factor: 2,
            num_sub_vectors: 16,
        }
    }

    #[test]
    fn usearch_build_minimal_allocation() {
        let dim = 128;
        let n = 100;
        let flat = vec![1.0f32; n * dim];
        let ids: Vec<[u8; 16]> = vec![[0; 16]; n];

        let (prev_count, prev_bytes) = alloc_counting::snapshot();
        let params = test_params();
        let graph = AnnIndex::build(dim, &flat, ids, &params).unwrap();
        std::hint::black_box(&graph);
        let (curr_count, curr_bytes) = alloc_counting::snapshot();

        let delta_bytes = curr_bytes - prev_bytes;
        let delta_count = curr_count - prev_count;
        eprintln!(
            "usearch_build: {n} vectors x {dim} dim, delta_bytes={delta_bytes}, delta_count={delta_count}"
        );
        assert!(
            delta_count < 500,
            "usearch should allocate far less than hnsw-rs (was ~22000), got {delta_count}",
        );
    }

    #[test]
    fn usearch_build_does_not_mutate_input_slice() {
        let dim = 16;
        let n = 10;
        let flat = vec![1.0f32; n * dim];
        let original = flat.clone();
        let ids = vec![[0; 16]; n];
        let params = test_params();
        let _graph = AnnIndex::build(dim, &flat, ids, &params).unwrap();
        assert_eq!(flat, original);
    }

    #[test]
    fn usearch_build_with_empty_input() {
        let dim = 64;
        let flat: Vec<f32> = Vec::new();
        let ids: Vec<[u8; 16]> = Vec::new();
        let params = test_params();

        let graph = AnnIndex::build(dim, &flat, ids, &params).unwrap();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        let mut out = Vec::new();
        graph.search(&[1.0f32; 64], 5, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn usearch_search_finds_inserted_entries() {
        let dim = 4;
        let n = 50;
        let mut flat = vec![0.0f32; n * dim];
        let mut ids = vec![[0; 16]; n];
        for i in 0..n {
            flat[i * dim] = (i + 1) as f32;
            ids[i][0..8].copy_from_slice(&(i as u64).to_le_bytes());
        }
        let params = test_params();
        let graph = AnnIndex::build(dim, &flat, ids, &params).unwrap();

        let query = vec![42.0, 0.0, 0.0, 0.0];
        let mut out = Vec::new();
        graph.search(&query, 5, &mut out);
        assert!(!out.is_empty(), "search should return results");
        for (_id, score) in &out {
            assert!(
                *score > 0.0,
                "cosine similarity should be positive for similar vectors"
            );
        }
    }

    #[test]
    fn builder_avoids_intermediate_flat_vec() {
        let dim = 128;
        let n = 100;
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0f32; dim]).collect();

        let params = test_params();
        let (prev_count, prev_bytes) = alloc_counting::snapshot();
        let mut builder = AnnIndexBuilder::new(dim, &params).unwrap();
        builder.reserve(n).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            builder.add(id, v);
        }
        let graph = builder.build().unwrap();
        std::hint::black_box(&graph);
        let (curr_count, curr_bytes) = alloc_counting::snapshot();

        let delta_bytes = curr_bytes - prev_bytes;
        let delta_count = curr_count - prev_count;
        let chunk_bytes = n * std::mem::size_of::<ChunkId>();
        let vector_bytes = n * dim * std::mem::size_of::<f32>();
        eprintln!(
            "builder: {n} vectors x {dim} dim, delta_bytes={delta_bytes}, delta_count={delta_count}, chunk_ids={chunk_bytes}, vectors_usearch_copies={vector_bytes}"
        );
        let expected_max = (vector_bytes + chunk_bytes) * 2;
        assert!(
            delta_bytes <= expected_max,
            "builder should allocate ~(vectors + chunk_ids)*2 (max {expected_max}), got {delta_bytes}",
        );
    }

    #[test]
    fn builder_produces_searchable_index() {
        let dim = 8;
        let n = 30;
        let params = test_params();
        let mut builder = AnnIndexBuilder::new(dim, &params).unwrap();
        builder.reserve(n).unwrap();
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[0] = (i + 1) as f32;
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            builder.add(id, &v);
        }
        let graph = builder.build().unwrap();

        let query = vec![15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = Vec::new();
        graph.search(&query, 3, &mut out);
        assert!(!out.is_empty());
    }
}
