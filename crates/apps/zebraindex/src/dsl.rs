use std::path::Path;

use anyhow::Result;
use clap::Subcommand;

use zti_dsl::render::dsl::{DslRenderer, render_files_only};
use zti_dsl::render::tree::AsciiTreeRenderer;
use zti_tree_sitter::{parse_kinds, parse_language};

#[derive(Subcommand)]
pub enum DslCommands {
    #[command(about = "Show the file tree with numeric IDs")]
    FileTree {
        #[arg(short, long, help = "Glob pattern to filter files")]
        path_glob: Option<String>,
    },
    #[command(about = "Show the DSL symbol map, sectioned by language")]
    ProjectMap {
        #[arg(
            short,
            long,
            help = "Restrict to one language (rs|ts|tsx|dart|sol). Omit to include all."
        )]
        language: Option<String>,
        #[arg(short, long, help = "Glob pattern to filter files")]
        path_glob: Option<String>,
        #[arg(
            short,
            long,
            help = "Filter by kinds: fn, method, struct, enum, class, const, module"
        )]
        kinds: Option<Vec<String>>,
        #[arg(short, long, default_value = "8000", help = "Max tokens")]
        max_tokens: usize,
    },
    #[command(about = "Trace dependency chains for a symbol")]
    DepTree {
        #[arg(short, long, help = "Symbol ID")]
        id: u32,
        #[arg(
            short = 'D',
            long,
            default_value = "callers",
            help = "Direction: callers or callees"
        )]
        direction: String,
        #[arg(short, long, default_value = "3", help = "Max depth")]
        depth: usize,
    },
    #[command(about = "Show the source code of a symbol")]
    SymbolBody {
        #[arg(short, long, help = "Symbol ID")]
        id: u32,
    },
    #[command(about = "Show the source code of multiple symbols")]
    SymbolBodies {
        #[arg(short, long, num_args(1..), help = "Symbol IDs")]
        ids: Vec<u32>,
    },
}

pub fn run_dsl(root: &Path, command: DslCommands) -> Result<()> {
    let canonical = root.canonicalize()?;
    let root_cow = canonical.to_string_lossy();

    let index = zti_dsl::build_index(&root_cow)?;
    tracing::info!(
        "{} symbols, {} edges, {} files",
        index.symbols.len(),
        index.edges.len(),
        index.files.len()
    );

    match command {
        DslCommands::FileTree { path_glob: _ } => {
            let file_indices: Vec<u16> = (0..index.files.len() as u16).collect();
            print!("{}", render_files_only(&index, &file_indices));
        }
        DslCommands::ProjectMap {
            language,
            path_glob: _,
            kinds,
            max_tokens,
        } => {
            let file_filter: Option<Vec<u16>> = language.as_ref().and_then(|l| {
                let lang = parse_language(l)?;
                Some(
                    index
                        .files
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| f.language == lang)
                        .map(|(i, _)| i as u16)
                        .collect(),
                )
            });
            let kind_filter: Option<Vec<zti_ts_core::types::Kind>> =
                kinds.as_ref().map(|k| parse_kinds(k));
            let renderer = DslRenderer::new(&index, max_tokens);
            print!(
                "{}",
                renderer.render(file_filter.as_deref(), kind_filter.as_deref())
            );
        }
        DslCommands::DepTree {
            id,
            direction,
            depth,
        } => {
            let renderer = AsciiTreeRenderer::new(&index);
            match direction.as_str() {
                "callers" => print!("{}", renderer.render_callers(id, depth)),
                "callees" => print!("{}", renderer.render_callees(id, depth, false)),
                _ => return Err(anyhow::anyhow!("direction must be 'callers' or 'callees'")),
            }
        }
        DslCommands::SymbolBody { id } => {
            let entries = zti_dsl::resolve_symbol_bodies(&index, &[id]);
            match entries.first() {
                Some(zti_common::dsl::SymbolBodyEntry::Ok {
                    kind_short,
                    symbol_id,
                    start_line,
                    end_line,
                    body,
                    ..
                }) => {
                    println!("{}#{} : {}-{}", kind_short, symbol_id, start_line, end_line);
                    println!("{}", body);
                }
                Some(zti_common::dsl::SymbolBodyEntry::Err { message, .. }) => {
                    return Err(anyhow::anyhow!("{}", message));
                }
                None => return Err(anyhow::anyhow!("Symbol {} not found", id)),
            }
        }
        DslCommands::SymbolBodies { ids } => {
            let entries = zti_dsl::resolve_symbol_bodies(&index, &ids);
            for entry in &entries {
                println!("{}\n---", entry);
            }
        }
    }

    Ok(())
}
