//! Broad per-file classification, persisted per chunk as one `u8` so the search
//! layer can hard-filter rows before distance/BM25 work.

use serde::{Deserialize, Serialize};

/// Persisted as `u8`; default `Source` keeps untyped rows useful.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileType {
    #[default]
    Source = 0,
    Test = 1,
    Config = 2,
    Doc = 3,
}

impl FileType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Test => "test",
            Self::Config => "config",
            Self::Doc => "doc",
        }
    }
}

impl From<FileType> for u8 {
    #[inline]
    fn from(file_type: FileType) -> Self {
        file_type as Self
    }
}

impl TryFrom<u8> for FileType {
    type Error = u8;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Source),
            1 => Ok(Self::Test),
            2 => Ok(Self::Config),
            3 => Ok(Self::Doc),
            other => Err(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FileType;

    #[test]
    fn file_type_round_trip() {
        let cases = [
            FileType::Source,
            FileType::Test,
            FileType::Config,
            FileType::Doc,
        ];
        for file_type in cases {
            let raw = u8::from(file_type);
            assert_eq!(FileType::try_from(raw), Ok(file_type));
        }
        assert_eq!(FileType::try_from(255), Err(255));
    }

    #[test]
    fn file_type_strings_are_stable() {
        assert_eq!(FileType::Source.as_str(), "source");
        assert_eq!(FileType::Test.as_str(), "test");
        assert_eq!(FileType::Config.as_str(), "config");
        assert_eq!(FileType::Doc.as_str(), "doc");
    }
}
