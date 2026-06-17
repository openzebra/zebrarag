use std::borrow::Cow;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::oneshot;
use zti_remote_embed::RemoteEmbedEngine;

use crate::{apply_prefix, EmbedEngine, Pooled};

/// Clip `s` to at most `max_bytes` at a valid UTF-8 char boundary. Never
/// panics: walks back from `max_bytes` to the nearest boundary, or returns
/// empty if none exists (only possible when `max_bytes` is within a multi-byte
/// sequence starting at byte 0).
fn truncate_to_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Multiplier on the provider's per-request item cap for the indexer's outer
/// embed loop. Set to 1: the outer loop should iterate at the provider's item
/// cap so progress is reported and RecordBatches flush incrementally — the
/// remote engine's `embed_texts` already pipelines HTTP requests internally.
/// Higher values (previously 8) made the indexer block on hundreds of HTTP
/// requests before reporting any progress, which looks like a hang.
const REMOTE_OUTER_BATCH_MULTIPLIER: usize = 1;

/// Cap on remote `context_length` for chunk-size decisions. Remote providers
/// report the underlying LLM's full context window (e.g. 128K), which is the
/// API's truncation limit, not an embedding-optimal chunk size. A 128K-token
/// embedding of a whole chapter is semantically near-useless and makes the
/// recursive chunker's DP pathological. 8 192 tokens (~32 KB) balances context
/// coverage with embedding precision.
const REMOTE_CHUNK_TOKENS_CAP: usize = 8192;

// This enum is stored behind Arc in daemon state; avoiding Box keeps one heap
// allocation and pointer chase out of every embedding call.
#[allow(clippy::large_enum_variant)]
pub enum AnyEmbedEngine {
    Local(EmbedEngine),
    Remote(RemoteEmbedEngine),
}

impl AnyEmbedEngine {
    #[inline]
    pub fn dim(&self) -> usize {
        match self {
            Self::Local(e) => e.dim(),
            Self::Remote(e) => e.dim(),
        }
    }

    #[inline]
    pub fn max_length(&self) -> usize {
        match self {
            Self::Local(e) => e.profile().max_length,
            Self::Remote(e) => e.max_length(),
        }
    }

    /// Token ceiling to use for *chunking decisions* (PDF packing budget,
    /// `adaptive_split` sizing). This differs from [`max_length`](Self::max_length):
    /// - **Local**: uses the curated `max_length` directly — real embedding
    ///   models (BERT 512, JinaBERT 8192) set this sensibly, so it is the right
    ///   chunk size.
    /// - **Remote**: caps `context_length` at [`REMOTE_CHUNK_TOKENS_CAP`],
    ///   because `context_length` is the LLM API's truncation limit, not an
    ///   embedding-optimal chunk size (a 128K-token chunk produces a useless
    ///   vector and pathological chunker DP).
    #[inline]
    pub fn chunk_max_tokens(&self) -> usize {
        match self {
            Self::Local(e) => e.profile().max_length,
            Self::Remote(e) => e.max_length().min(REMOTE_CHUNK_TOKENS_CAP),
        }
    }

    /// Conservative bytes-per-token estimate for chunk-size decisions. The
    /// true ratio varies by content: ~4-5 for prose, ~3.5 for code, but only
    /// ~2.1-2.45 for dense PDF math text (LaTeX-derived fonts, Unicode
    /// symbols, frequent single-char tokens). Measured from OpenRouter HTTP
    /// 400s: a 24576-byte chunk at ratio 3 decoded to 10047-11555 tokens
    /// (~2.1-2.4 B/tok).
    /// - **Local**: returns 4 — the local tokenizer truncates precisely if
    ///   the estimate is slightly wrong, so a generous ratio avoids
    ///   over-splitting.
    /// - **Remote**: returns 2 — there is no client-side tokenizer, so a too-
    ///   large chunk is a hard HTTP 400 with no truncation fallback. The
    ///   send-boundary cap (`cap_remote_texts`) is the final safety net for
    ///   any residual overshoot from packing/overlap.
    #[inline]
    pub const fn chars_per_token(&self) -> usize {
        match self {
            Self::Local(_) => 8,
            Self::Remote(_) => 1,
        }
    }

    /// Hard per-item byte cap for remote batch sends. Clips each input to
    /// `chunk_max_tokens × chars_per_token` at a UTF-8 char boundary before
    /// sending to the provider. This is the final safety net: even if
    /// `pack_pdf_pages` overshoots its budget or `adaptive_split` mis-estimates
    /// the byte→token ratio, no item can exceed the model's token limit.
    ///
    /// Zero-copy: returns borrowed prefix slices; only a `Vec<&str>` of
    /// pointers is allocated. Correctly-sized chunks pass untouched.
    fn cap_remote_texts<'t>(&self, texts: &[&'t str]) -> Vec<&'t str> {
        let max_bytes = self
            .chunk_max_tokens()
            .saturating_mul(self.chars_per_token());
        let mut clipped = 0usize;
        let capped = texts
            .iter()
            .map(|s| {
                let t = truncate_to_char_boundary(s, max_bytes);
                clipped += usize::from(t.len() < s.len());
                t
            })
            .collect::<Vec<_>>();
        if clipped > 0 {
            tracing::warn!(
                clipped,
                max_bytes,
                "remote embed: clipped oversized chunk(s) to byte cap"
            );
        }
        capped
    }

    #[inline]
    pub const fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }

    /// Submit raw texts to the worker for tokenization + forward pass.
    /// Returns immediately — no synchronous CPU work on the tokio reactor.
    pub fn submit_texts_pooled(&self, texts: &[&str]) -> Result<oneshot::Receiver<Result<Pooled>>> {
        match self {
            Self::Local(e) => {
                if let Some(prefix) = e.profile().passage_prefix.as_deref() {
                    let owned: Arc<[String]> =
                        texts.iter().map(|t| format!("{prefix}{t}")).collect();
                    e.submit_texts(owned)
                } else {
                    let owned: Arc<[String]> = texts.iter().map(|s| (*s).to_string()).collect();
                    e.submit_texts(owned)
                }
            }
            Self::Remote(_) => anyhow::bail!("pipelined submit not available for remote engines"),
        }
    }

    pub fn model_id_str(&self) -> &str {
        match self {
            Self::Local(e) => e.profile().model_id.as_str(),
            Self::Remote(e) => e.model_id(),
        }
    }

    pub fn persisted_model_id(&self) -> Cow<'_, str> {
        match self {
            Self::Local(e) => Cow::Borrowed(e.profile().model_id.as_str()),
            Self::Remote(e) => {
                Cow::Owned(format!("{}{}", e.provider().model_prefix(), e.model_id()))
            }
        }
    }

    pub fn hardware(&self) -> Option<&zti_hw::Hardware> {
        match self {
            Self::Local(e) => Some(e.hardware()),
            Self::Remote(_) => None,
        }
    }

    pub fn device_with_hardware(&self, hardware: &zti_hw::Hardware) -> Result<candle_core::Device> {
        match self {
            Self::Local(e) => e.device(),
            Self::Remote(_) => Ok(zti_hw::candle_device(hardware)),
        }
    }

    pub fn recommended_batch_size(&self) -> usize {
        match self {
            Self::Local(e) => e.recommended_batch_size(),
            Self::Remote(e) => e
                .provider()
                .max_batch_items()
                .saturating_mul(REMOTE_OUTER_BATCH_MULTIPLIER)
                .max(1),
        }
    }

    /// Unified passage embedding that returns `Pooled` directly, bypassing
    /// the `Vec<Vec<f32>>` roundtrip. Prefix application produces `String`s
    /// directly for the worker channel — no intermediate Cow layer.
    pub async fn embed_texts_pooled_async(&self, texts: &[&str]) -> Result<Pooled> {
        let dim = self.dim();
        match self {
            Self::Local(e) => {
                if let Some(prefix) = e.profile().passage_prefix.as_deref() {
                    let owned: Arc<[String]> =
                        texts.iter().map(|t| format!("{prefix}{t}")).collect();
                    e.submit_texts(owned)?
                        .await
                        .map_err(|_| anyhow::anyhow!("embed worker dropped without replying"))?
                } else {
                    e.embed_batch_pooled_async(texts).await
                }
            }
            Self::Remote(e) => {
                let capped = self.cap_remote_texts(texts);
                let rows = e.embed_texts(&capped).await?;
                let batch = rows.len();
                let mut data = Vec::with_capacity(batch.saturating_mul(dim));
                for row in rows {
                    data.extend(row);
                }
                Ok(Pooled { data, dim, batch })
            }
        }
    }

    /// Unified passage embedding: local applies the model's passage prefix, remote sends raw text.
    pub async fn embed_texts_async(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Local(e) => {
                let prefixed: Vec<Cow<'_, str>> = texts
                    .iter()
                    .map(|text| apply_prefix(text, e.profile().passage_prefix.as_deref()))
                    .collect();
                let refs: Vec<&str> = prefixed.iter().map(|text| text.as_ref()).collect();
                e.embed_batch_async(&refs).await
            }
            Self::Remote(e) => {
                let capped = self.cap_remote_texts(texts);
                e.embed_texts(&capped).await
            }
        }
    }

    pub async fn embed_query_async(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Local(e) => e.embed_query_async(text).await,
            Self::Remote(e) => e.embed_query(text).await,
        }
    }

    pub async fn embed_passage_async(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Local(e) => e.embed_passage_async(text).await,
            Self::Remote(e) => e.embed_passage(text).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::truncate_to_char_boundary;

    #[test]
    fn truncate_passthrough_when_within_budget() {
        assert_eq!(truncate_to_char_boundary("hello", 10), "hello");
        assert_eq!(truncate_to_char_boundary("hello", 5), "hello");
        assert_eq!(truncate_to_char_boundary("", 0), "");
    }

    #[test]
    fn truncate_ascii_at_exact_byte() {
        assert_eq!(truncate_to_char_boundary("hello world", 5), "hello");
    }

    #[test]
    fn truncate_never_splits_multibyte_char() {
        // \u{1F600} (😀) is 4 bytes. Capping at byte 1-3 must walk back to 0.
        let s = "ab\u{1F600}cd";
        assert_eq!(truncate_to_char_boundary(s, 3), "ab");
        assert_eq!(truncate_to_char_boundary(s, 2), "ab");
        assert_eq!(truncate_to_char_boundary(s, 1), "a");
        assert_eq!(truncate_to_char_boundary(s, 0), "");
        // Capping just before the emoji keeps it out.
        assert_eq!(truncate_to_char_boundary(s, 4), "ab");
        // Capping at the emoji's last byte includes it.
        assert_eq!(truncate_to_char_boundary(s, 6), "ab\u{1F600}");
    }

    #[test]
    fn truncate_cjk_boundary() {
        // Each CJK char is 3 bytes in UTF-8.
        let s = "\u{4E2D}\u{6587}"; // 中文
        assert_eq!(s.len(), 6);
        assert_eq!(truncate_to_char_boundary(s, 5), "\u{4E2D}"); // 3 bytes
        assert_eq!(truncate_to_char_boundary(s, 3), "\u{4E2D}");
        assert_eq!(truncate_to_char_boundary(s, 6), s);
    }
}
