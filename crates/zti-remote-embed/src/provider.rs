#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteProvider {
    OpenRouter,
    OpenAI,
    XAI,
    Alibaba,
}

impl RemoteProvider {
    /// Every supported provider, for iteration (model-id resolution, menus).
    pub const ALL: &'static [Self] = &[Self::OpenRouter, Self::OpenAI, Self::XAI, Self::Alibaba];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenRouter => "openrouter",
            Self::OpenAI => "openai",
            Self::XAI => "xai",
            Self::Alibaba => "alibaba",
        }
    }

    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OpenRouter => "OpenRouter",
            Self::OpenAI => "OpenAI",
            Self::XAI => "xAI",
            Self::Alibaba => "Alibaba",
        }
    }

    #[inline]
    pub const fn env_var(self) -> &'static str {
        match self {
            Self::OpenRouter => "ZEBRA_OPENROUTER_KEY",
            Self::OpenAI => "ZEBRA_OPENAI_KEY",
            Self::XAI => "ZEBRA_XAI_KEY",
            Self::Alibaba => "ZEBRA_DASHSCOPE_KEY",
        }
    }

    /// Prefix that tags a model id as belonging to this provider, e.g.
    /// `openai:text-embedding-3-small`.
    #[inline]
    pub const fn model_prefix(self) -> &'static str {
        match self {
            Self::OpenRouter => "openrouter:",
            Self::OpenAI => "openai:",
            Self::XAI => "xai:",
            Self::Alibaba => "alibaba:",
        }
    }

    #[inline]
    pub const fn base_url(self) -> &'static str {
        match self {
            Self::OpenRouter => "https://openrouter.ai/api/v1",
            Self::OpenAI => "https://api.openai.com/v1",
            Self::XAI => "https://api.x.ai/v1",
            Self::Alibaba => "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        }
    }

    /// Query string that restricts the provider's `/models` listing to
    /// embedding models server-side. Empty when the provider needs no filter
    /// (the listing is then filtered client-side — see
    /// [`requires_client_side_embedding_filter`](Self::requires_client_side_embedding_filter)).
    #[inline]
    pub const fn models_query(self) -> &'static str {
        match self {
            Self::OpenRouter => "output_modalities=embeddings",
            Self::OpenAI | Self::XAI | Self::Alibaba => "",
        }
    }

    /// Path GET'd to validate an API key; a 401 there means a bad key.
    /// OpenRouter exposes a free `/key` endpoint; the OpenAI-compatible
    /// providers reuse `/models`.
    #[inline]
    pub const fn validate_path(self) -> &'static str {
        match self {
            Self::OpenRouter => "/key",
            Self::OpenAI | Self::XAI | Self::Alibaba => "/models",
        }
    }

    /// Extra request headers a provider requires. OpenRouter asks for an
    /// attribution referer/title; the others need none.
    #[inline]
    pub const fn extra_headers(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::OpenRouter => &[
                ("http-referer", "https://github.com/hicaru/zebra_tree_indexer"),
                ("x-title", "zebraindex"),
            ],
            Self::OpenAI | Self::XAI | Self::Alibaba => &[],
        }
    }

    /// Whether the `/models` listing returns non-embedding models that must be
    /// filtered client-side (true unless the provider filters server-side).
    #[inline]
    pub const fn requires_client_side_embedding_filter(self) -> bool {
        match self {
            Self::OpenRouter => false,
            Self::OpenAI | Self::XAI | Self::Alibaba => true,
        }
    }

    /// Conservative item-count cap for one provider embeddings request.
    /// The remote engine also applies a char-budget split; this outer count
    /// cap avoids oversized JSON arrays on providers with element limits.
    #[inline]
    pub const fn max_batch_items(self) -> usize {
        match self {
            Self::OpenRouter | Self::OpenAI => 512,
            // DashScope caps `input` at 25 items; xAI follows the smaller bound too.
            Self::XAI | Self::Alibaba => 25,
        }
    }

    /// Resolve the provider that owns a prefixed model id, returning the
    /// provider and the bare model id (prefix stripped). `None` for a local
    /// (unprefixed) model id.
    #[inline]
    pub fn from_model_id(model_id: &str) -> Option<(Self, &str)> {
        Self::ALL
            .iter()
            .find_map(|p| model_id.strip_prefix(p.model_prefix()).map(|rest| (*p, rest)))
    }
}

/// Heuristic for OpenAI-compatible `/models` listings that don't tag modality:
/// keep ids that look like embedding models.
#[inline]
pub fn is_embedding_model(id: &str) -> bool {
    id.to_ascii_lowercase().contains("embed")
}

impl TryFrom<&str> for RemoteProvider {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::ALL
            .iter()
            .copied()
            .find(|p| p.as_str() == value)
            .ok_or_else(|| anyhow::anyhow!("unsupported remote provider: {value}"))
    }
}
