use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

pub fn chunks_schema(dim: usize) -> Schema {
    let fields = vec![
        Field::new("chunk_id", DataType::FixedSizeBinary(16), false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("language", DataType::Utf8, false),
        Field::new("symbol_qualified", DataType::Utf8, true),
        Field::new("symbol_kind", DataType::Utf8, true),
        Field::new("parent_qualified", DataType::Utf8, true),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("turbo_code", DataType::Binary, true),
        Field::new("indexed_at_ns", DataType::UInt64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dim as i32,
            ),
            false,
        ),
    ];
    Schema::new(fields)
}

pub fn files_schema() -> Schema {
    let fields = vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("blake3", DataType::FixedSizeBinary(32), false),
        Field::new("mtime_ns", DataType::UInt64, false),
        Field::new("size_bytes", DataType::UInt64, false),
        Field::new("language", DataType::Utf8, false),
        Field::new(
            "chunk_ids",
            DataType::List(Arc::new(Field::new("item", DataType::FixedSizeBinary(16), false))),
            false,
        ),
        Field::new("indexed_at_ns", DataType::UInt64, false),
    ];
    Schema::new(fields)
}

pub fn projects_schema() -> Schema {
    let fields = vec![
        Field::new("project_id", DataType::FixedSizeBinary(32), false),
        Field::new("root_path", DataType::Utf8, false),
        Field::new("languages", DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))), false),
        Field::new("model_id", DataType::Utf8, false),
        Field::new("model_dim", DataType::UInt32, false),
        Field::new("total_chunks", DataType::UInt64, false),
        Field::new("total_files", DataType::UInt64, false),
        Field::new("last_indexed_ns", DataType::UInt64, false),
        Field::new("created_at_ns", DataType::UInt64, false),
    ];
    Schema::new(fields)
}
