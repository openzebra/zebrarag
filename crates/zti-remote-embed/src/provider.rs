#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteProvider {
    OpenRouter,
}

impl RemoteProvider {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenRouter => "openrouter",
        }
    }

    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OpenRouter => "OpenRouter",
        }
    }

    #[inline]
    pub const fn env_var(self) -> &'static str {
        match self {
            Self::OpenRouter => "ZEBRA_OPENROUTER_KEY",
        }
    }

    #[inline]
    pub const fn model_prefix(self) -> &'static str {
        match self {
            Self::OpenRouter => "openrouter:",
        }
    }

    #[inline]
    pub const fn base_url(self) -> &'static str {
        match self {
            Self::OpenRouter => "https://openrouter.ai/api/v1",
        }
    }

    /// Conservative item-count cap for one provider embeddings request.
    /// The remote engine also applies a char-budget split; this outer count
    /// cap avoids oversized JSON arrays on providers with element limits.
    #[inline]
    pub const fn max_batch_items(self) -> usize {
        match self {
            Self::OpenRouter => 512,
        }
    }
}

impl TryFrom<&str> for RemoteProvider {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "openrouter" => Ok(Self::OpenRouter),
            other => anyhow::bail!("unsupported remote provider: {other}"),
        }
    }
}
