use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use zti_dsl::render::dsl::{DslRenderer, render_files_only};
use zti_dsl::render::tree::AsciiTreeRenderer;
use zti_tree_sitter::Language;

#[derive(Parser)]
#[command(name = "zebra-dsl", version, about = "DSL graph dump for debugging")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, help = "Project root path")]
    root: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Show the file tree with numeric IDs")]
    FileTree {
        #[arg(short, long, help = "Glob pattern to filter files")]
        path_glob: Option<String>,
    },
    #[command(about = "Show the DSL symbol map for one language")]
    ProjectMap {
        #[arg(short, long, help = "Language: rs|rust, ts|tsx|typescript, dart, sol|solidity")]
        language: String,
        #[arg(short, long, help = "Glob pattern to filter files")]
        path_glob: Option<String>,
        #[arg(short, long, help = "Filter by kinds: fn, method, struct, enum, class, const, module")]
        kinds: Option<Vec<String>>,
        #[arg(short, long, default_value = "8000", help = "Max tokens")]
        max_tokens: usize,
    },
    #[command(about = "Trace dependency chains for a symbol")]
    DepTree {
        #[arg(short, long, help = "Symbol ID")]
        id: u32,
        #[arg(short = 'D', long, default_value = "callers", help = "Direction: callers or callees")]
        direction: String,
        #[arg(short, long, default_value = "3", help = "Max depth")]
        depth: usize,
    },
    #[command(about = "Show the source code of a symbol")]
    SymbolBody {
        #[arg(short, long, help = "Symbol ID")]
        id: u32,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();
    let root_str = cli.root.canonicalize()?.to_string_lossy().to_string();

    let index = zti_dsl::build_index(&root_str)?;
    tracing::info!("{} symbols, {} edges, {} files", index.symbols.len(), index.edges.len(), index.files.len());

    match cli.command {
        Commands::FileTree { path_glob: _ } => {
            let file_indices: Vec<u16> = (0..index.files.len() as u16).collect();
            print!("{}", render_files_only(&index, &file_indices));
        }
        Commands::ProjectMap { language, path_glob: _, kinds, max_tokens } => {
            let lang = parse_language(&language);
            let file_filter: Option<Vec<u16>> = lang.map(|l| {
                index.files.iter().enumerate()
                    .filter(|(_, f)| f.language == l)
                    .map(|(i, _)| i as u16)
                    .collect()
            });
            let kind_filter: Option<Vec<zti_ts_core::types::Kind>> = kinds.as_ref().map(|k| parse_kinds(k));
            let renderer = DslRenderer::new(&index, max_tokens);
            print!("{}", renderer.render(file_filter.as_deref(), kind_filter.as_deref()));
        }
        Commands::DepTree { id, direction, depth } => {
            let renderer = AsciiTreeRenderer::new(&index);
            match direction.as_str() {
                "callers" => print!("{}", renderer.render_callers(id, depth)),
                "callees" => print!("{}", renderer.render_callees(id, depth)),
                _ => return Err(anyhow::anyhow!("direction must be 'callers' or 'callees'")),
            }
        }
        Commands::SymbolBody { id } => {
            let sym = index.symbols.get(id as usize)
                .ok_or_else(|| anyhow::anyhow!("Symbol {} not found", id))?;
            let file = index.files.get(sym.file_idx as usize)
                .ok_or_else(|| anyhow::anyhow!("File not found for symbol {}", id))?;
            let content = std::fs::read_to_string(&file.path)?;
            let lines: Vec<&str> = content.lines().collect();
            let start = (sym.line as usize).saturating_sub(1);
            let end = (sym.end_line as usize).min(lines.len());
            println!("// File: {} | Lines: {}-{}", file.path, sym.line, sym.end_line);
            println!("{}", lines[start..end].join("\n"));
        }
    }

    Ok(())
}

fn parse_language(s: &str) -> Option<Language> {
    match s.to_ascii_lowercase().as_str() {
        "rs" | "rust" => Some(Language::Rust),
        "ts" | "tsx" | "typescript" => Some(Language::Ts),
        "dart" => Some(Language::Dart),
        "sol" | "solidity" => Some(Language::Solidity),
        _ => None,
    }
}

fn parse_kinds(kinds: &[String]) -> Vec<zti_ts_core::types::Kind> {
    kinds.iter().filter_map(|k| zti_ts_core::types::Kind::from_str_lossy(k)).collect()
}
