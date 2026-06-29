use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolBodyEntry {
    Ok {
        symbol_id: u32,
        kind_short: Cow<'static, str>,
        start_line: u32,
        end_line: u32,
        body: String,
    },
    Err {
        symbol_id: u32,
        message: String,
    },
}

impl fmt::Display for SymbolBodyEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok {
                symbol_id,
                kind_short,
                start_line,
                end_line,
                body,
            } => write!(
                f,
                "{}#{} : {}-{}\n{}",
                kind_short, symbol_id, start_line, end_line, body
            ),
            Self::Err { symbol_id, message } => {
                write!(f, "#{} : ERROR: {}", symbol_id, message)
            }
        }
    }
}
