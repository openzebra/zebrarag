//! Minimal BERT wrapper around candle's public `BertEncoder` that fixes candle
//! 0.10's reduced-precision attention-mask bug.
//!
//! candle's `get_extended_attention_mask` builds the additive mask as
//! `(1 - mask) * f32::MIN`. In F16/BF16, `f32::MIN` overflows to `-inf`, so a
//! valid token's `(1 - 1) * -inf = 0 * -inf = NaN`, which poisons the whole
//! output (100% NaN embeddings). We reuse candle's `BertEncoder` unchanged and
//! reimplement only the (private) embeddings layer and the mask, using a finite
//! fill so `0 * fill = 0`.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module, VarBuilder, embedding, layer_norm};
use candle_transformers::models::bert::{BertEncoder, Config};

/// Classic BERT mask fill: finite in every dtype (|value| << F16 max 65504), and
/// `exp(-1e4) ~= 0` after softmax, so masked positions contribute nothing.
const MASK_FILL: f32 = -1e4;

struct Embeddings {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    layer_norm: LayerNorm,
}

impl Embeddings {
    fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        Ok(Self {
            word: embedding(c.vocab_size, c.hidden_size, vb.pp("word_embeddings"))?,
            position: embedding(
                c.max_position_embeddings,
                c.hidden_size,
                vb.pp("position_embeddings"),
            )?,
            token_type: embedding(
                c.type_vocab_size,
                c.hidden_size,
                vb.pp("token_type_embeddings"),
            )?,
            layer_norm: layer_norm(c.hidden_size, c.layer_norm_eps, vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let e = (self.word.forward(input_ids)? + self.token_type.forward(token_type_ids)?)?;
        // candle uses 0..seq_len absolute positions; match it for index/search parity.
        let pos = Tensor::arange(0u32, seq_len as u32, input_ids.device())?;
        let e = e.broadcast_add(&self.position.forward(&pos)?)?;
        self.layer_norm.forward(&e)
    }
}

pub struct BertModel {
    embeddings: Embeddings,
    encoder: BertEncoder,
}

impl BertModel {
    pub fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        // Same fallback as candle: bare names first, then `{model_type}.` prefix
        // (e5 / XLM-R safetensors). Preserves existing load behavior exactly.
        let (embeddings, encoder) = match (
            Embeddings::load(vb.pp("embeddings"), c),
            BertEncoder::load(vb.pp("encoder"), c),
        ) {
            (Ok(emb), Ok(enc)) => (emb, enc),
            (Err(err), _) | (_, Err(err)) => {
                let Some(mt) = &c.model_type else {
                    return Err(err);
                };
                match (
                    Embeddings::load(vb.pp(format!("{mt}.embeddings")), c),
                    BertEncoder::load(vb.pp(format!("{mt}.encoder")), c),
                ) {
                    (Ok(emb), Ok(enc)) => (emb, enc),
                    _ => return Err(err),
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
        })
    }

    /// Signature identical to `candle_transformers::models::bert::BertModel::forward`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden = self.embeddings.forward(input_ids, token_type_ids)?;
        let dtype = hidden.dtype();
        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => input_ids.ones_like()?,
        };
        let ext = extended_attention_mask(&mask, dtype)?;
        self.encoder.forward(&hidden, &ext)
    }
}

/// candle's `get_extended_attention_mask`, fixed: a finite fill so a valid
/// token's `(1 - 1) * fill = 0` (never `0 * -inf = NaN`).
pub(crate) fn extended_attention_mask(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let mask = match mask.rank() {
        3 => mask.unsqueeze(1)?,
        2 => mask.unsqueeze(1)?.unsqueeze(1)?,
        r => candle_core::bail!("attention_mask rank {r} unsupported"),
    };
    let mask = mask.to_dtype(dtype)?;
    let fill = Tensor::try_from(MASK_FILL)?
        .to_device(mask.device())?
        .to_dtype(dtype)?;
    // (1 - mask) * fill: 0 for valid tokens, `fill` for padding.
    (mask.ones_like()? - &mask)?.broadcast_mul(&fill)
}
