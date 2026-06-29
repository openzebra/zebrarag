use std::borrow::Cow;

use rustc_hash::FxHashMap;

use zrag_common::LineIndex;
use zrag_common::dsl::SymbolBodyEntry;

use crate::chunking::find_doc_start_line;
use crate::model::ProjectIndex;

pub fn resolve_symbol_bodies(index: &ProjectIndex, symbol_ids: &[u32]) -> Vec<SymbolBodyEntry> {
    let mut entries = Vec::with_capacity(symbol_ids.len());
    let mut file_cache: FxHashMap<u16, Result<(String, LineIndex), String>> =
        FxHashMap::with_capacity_and_hasher(symbol_ids.len(), rustc_hash::FxBuildHasher);

    for &id in symbol_ids {
        let sym = match index.symbols.get(id as usize) {
            Some(s) => s,
            None => {
                entries.push(SymbolBodyEntry::Err {
                    symbol_id: id,
                    message: format!("Symbol {} not found", id),
                });
                continue;
            }
        };

        let file = match index.files.get(sym.file_idx as usize) {
            Some(f) => f,
            None => {
                entries.push(SymbolBodyEntry::Err {
                    symbol_id: id,
                    message: format!("File for symbol {} not found", id),
                });
                continue;
            }
        };

        let content = file_cache.entry(sym.file_idx).or_insert_with(|| {
            std::fs::read_to_string(&file.path)
                .map(|s| {
                    let idx = LineIndex::new(&s);
                    (s, idx)
                })
                .map_err(|err| format!("Failed to read {}: {}", file.path, err))
        });

        match &*content {
            Ok((c, line_index)) => {
                let doc_start = if sym.doc.is_some() {
                    find_doc_start_line(c, sym.line, line_index)
                } else {
                    sym.line
                };
                let range = line_index.byte_range(doc_start, sym.end_line);
                entries.push(SymbolBodyEntry::Ok {
                    symbol_id: id,
                    kind_short: Cow::Borrowed(sym.kind.short()),
                    start_line: doc_start,
                    end_line: sym.end_line,
                    body: c[range].to_owned(),
                });
            }
            Err(msg) => {
                entries.push(SymbolBodyEntry::Err {
                    symbol_id: id,
                    message: msg.clone(),
                });
            }
        }
    }

    entries
}
