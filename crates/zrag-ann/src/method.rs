use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMethod {
    Flat,
    Usearch,
    IvfHnswSq,
    IvfHnswPq,
    IvfPq,
    IvfSq,
    IvfRq,
    TurboQuant,
}

impl SearchMethod {
    pub const ALL: [Self; 8] = [
        Self::IvfHnswSq,
        Self::IvfHnswPq,
        Self::IvfPq,
        Self::IvfSq,
        Self::IvfRq,
        Self::Usearch,
        Self::Flat,
        Self::TurboQuant,
    ];

    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Usearch => "usearch",
            Self::IvfHnswSq => "ivf_hnsw_sq",
            Self::IvfHnswPq => "ivf_hnsw_pq",
            Self::IvfPq => "ivf_pq",
            Self::IvfSq => "ivf_sq",
            Self::IvfRq => "ivf_rq",
            Self::TurboQuant => "turbo_quant",
        }
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::Usearch => "Usearch (HNSW)",
            Self::IvfHnswSq => "IVF_HNSW_SQ",
            Self::IvfHnswPq => "IVF_HNSW_PQ",
            Self::IvfPq => "IVF_PQ",
            Self::IvfSq => "IVF_SQ",
            Self::IvfRq => "IVF_RQ",
            Self::TurboQuant => "TurboQuant",
        }
    }

    #[inline]
    pub fn description(self) -> &'static str {
        match self {
            Self::IvfHnswSq => {
                "IVF + HNSW graph + scalar quantization. Best balance of speed, recall, and memory."
            }
            Self::IvfHnswPq => {
                "IVF + HNSW graph + product quantization. Lower memory than SQ, slightly lower recall."
            }
            Self::IvfPq => "IVF + product quantization. No graph. Fast build, moderate recall.",
            Self::IvfSq => {
                "IVF + scalar quantization. No graph. Compact, good for mid-size projects."
            }
            Self::IvfRq => "IVF + RabitQ quantization. High compression ratio.",
            Self::Usearch => {
                "In-memory HNSW graph. Full F32 precision. Highest recall, requires RAM."
            }
            Self::Flat => "Brute-force cosine scan. Perfect recall. Only for small projects.",
            Self::TurboQuant => {
                "Bitpolar rerank-only scan. No vector index. Ultra-fast, lowest memory."
            }
        }
    }

    #[inline]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "flat" => Some(Self::Flat),
            "usearch" => Some(Self::Usearch),
            "ivf_hnsw_sq" => Some(Self::IvfHnswSq),
            "ivf_hnsw_pq" => Some(Self::IvfHnswPq),
            "ivf_pq" => Some(Self::IvfPq),
            "ivf_sq" => Some(Self::IvfSq),
            "ivf_rq" => Some(Self::IvfRq),
            "turbo_quant" => Some(Self::TurboQuant),
            _ => None,
        }
    }

    #[inline]
    pub fn is_lancedb_index(self) -> bool {
        matches!(
            self,
            Self::IvfHnswSq | Self::IvfHnswPq | Self::IvfPq | Self::IvfSq | Self::IvfRq
        )
    }

    #[inline]
    pub fn stats(self) -> MethodStats {
        match self {
            Self::IvfHnswSq => MethodStats {
                accuracy: 82,
                search_speed: 95,
                build_speed: 60,
                compression: 80,
                best_for: "Large projects (>10K chunks)",
                storage_note: "~25% of raw vectors",
                ram_note: "Low (disk-backed)",
                params: &[
                    ("Partitions", "sqrt(n) clamped 4..256"),
                    ("HNSW edges", "16 (m)"),
                    ("ef_construct", "200"),
                    ("ef_search", "100"),
                    ("nprobes", "max(partitions/10, 4)"),
                    ("refine", "2x"),
                    ("quantization", "Scalar (SQ8)"),
                ],
            },
            Self::IvfHnswPq => MethodStats {
                accuracy: 70,
                search_speed: 88,
                build_speed: 40,
                compression: 90,
                best_for: "Very large projects, low RAM",
                storage_note: "~6% of raw vectors",
                ram_note: "Low (disk-backed)",
                params: &[
                    ("Partitions", "sqrt(n) clamped 4..256"),
                    ("HNSW edges", "16 (m)"),
                    ("ef_construct", "200"),
                    ("ef_search", "100"),
                    ("sub_vectors", "dim/8"),
                    ("quantization", "Product (PQ)"),
                ],
            },
            Self::IvfPq => MethodStats {
                accuracy: 65,
                search_speed: 80,
                build_speed: 70,
                compression: 90,
                best_for: "Large projects, fast build",
                storage_note: "~6% of raw vectors",
                ram_note: "Low (disk-backed)",
                params: &[
                    ("Partitions", "sqrt(n) clamped 4..256"),
                    ("nprobes", "max(partitions/10, 4)"),
                    ("sub_vectors", "dim/8"),
                    ("quantization", "Product (PQ)"),
                ],
            },
            Self::IvfSq => MethodStats {
                accuracy: 75,
                search_speed: 82,
                build_speed: 75,
                compression: 80,
                best_for: "Mid-to-large projects",
                storage_note: "~25% of raw vectors",
                ram_note: "Low (disk-backed)",
                params: &[
                    ("Partitions", "sqrt(n) clamped 4..256"),
                    ("nprobes", "max(partitions/10, 4)"),
                    ("quantization", "Scalar (SQ8)"),
                ],
            },
            Self::IvfRq => MethodStats {
                accuracy: 78,
                search_speed: 85,
                build_speed: 50,
                compression: 95,
                best_for: "Max compression experiments",
                storage_note: "~3% of raw vectors",
                ram_note: "Low (disk-backed)",
                params: &[
                    ("Partitions", "sqrt(n) clamped 4..256"),
                    ("nprobes", "max(partitions/10, 4)"),
                    ("quantization", "RabitQ"),
                ],
            },
            Self::Usearch => MethodStats {
                accuracy: 98,
                search_speed: 90,
                build_speed: 70,
                compression: 0,
                best_for: "Mid-size projects (1K-10K)",
                storage_note: "Full vectors in RAM",
                ram_note: "High (n * dim * 4 bytes)",
                params: &[
                    ("HNSW edges", "16 (m)"),
                    ("ef_construct", "200"),
                    ("ef_search", "100"),
                    ("quantization", "None (F32)"),
                    ("storage", "In-memory"),
                ],
            },
            Self::Flat => MethodStats {
                accuracy: 100,
                search_speed: 30,
                build_speed: 100,
                compression: 0,
                best_for: "Small projects (<1K chunks)",
                storage_note: "Full vectors on disk",
                ram_note: "Scanned on query",
                params: &[("quantization", "None (F32)"), ("storage", "LanceDB scan")],
            },
            Self::TurboQuant => MethodStats {
                accuracy: 40,
                search_speed: 99,
                build_speed: 100,
                compression: 100,
                best_for: "Ultra-fast, low accuracy OK",
                storage_note: "~1% of raw vectors",
                ram_note: "Minimal",
                params: &[
                    ("bits", "3 (bitpolar)"),
                    ("projections", "2 * dim"),
                    ("quantization", "Bitpolar (ICLR 2026)"),
                    ("storage", "Compact codes only"),
                ],
            },
        }
    }
}

pub struct MethodStats {
    pub accuracy: u8,
    pub search_speed: u8,
    pub build_speed: u8,
    pub compression: u8,
    pub best_for: &'static str,
    pub storage_note: &'static str,
    pub ram_note: &'static str,
    pub params: &'static [(&'static str, &'static str)],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchParams {
    pub method: SearchMethod,
    pub indexed_chunks: u64,
    pub m: u32,
    pub ef_construction: u32,
    pub ef_search: u32,
    pub num_partitions: u32,
    pub nprobes: u32,
    pub refine_factor: u32,
    #[serde(default)]
    pub num_sub_vectors: u32,
}
