use anyhow::Result;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Tokenized> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let mask = encoding.get_attention_mask().to_vec();
        Ok(Tokenized { ids, mask })
    }

    /// Read the tokenizer's truncation `max_length` if configured.
    pub fn truncation_max_length(&self) -> Option<usize> {
        self.inner.get_truncation().map(|t| t.max_length)
    }
}

pub struct Tokenized {
    pub ids: Vec<u32>,
    pub mask: Vec<u32>,
}
