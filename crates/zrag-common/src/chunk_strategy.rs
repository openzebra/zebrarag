use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStrategy {
    Symbol = 0,
    Recursive = 1,
}

impl From<u8> for ChunkStrategy {
    #[inline]
    fn from(val: u8) -> Self {
        match val {
            1 => Self::Recursive,
            _ => Self::Symbol,
        }
    }
}

#[cfg(test)]
mod tests_indexing {
    use super::*;

    #[test]
    fn test_chunk_strategy_conversion() {
        assert_eq!(ChunkStrategy::Symbol as u8, 0);
        assert_eq!(ChunkStrategy::Recursive as u8, 1);

        assert_eq!(ChunkStrategy::from(0), ChunkStrategy::Symbol);
        assert_eq!(ChunkStrategy::from(1), ChunkStrategy::Recursive);
        assert_eq!(ChunkStrategy::from(2), ChunkStrategy::Symbol);
        assert_eq!(ChunkStrategy::from(255), ChunkStrategy::Symbol);
    }
}
