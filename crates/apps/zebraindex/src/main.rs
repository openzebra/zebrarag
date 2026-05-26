use std::borrow::Cow;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

mod cli;
mod dsl;
mod mcp;
mod tui;

#[derive(Parser)]
#[command(name = "zebraindex", version, about = "Zebra semantic code indexer")]
struct TopLevel {
    #[arg(long, help = "Run as MCP server (stdio)")]
    mcp: bool,

    #[arg(short, long, global = true)]
    model: Option<String>,

    #[arg(long, global = true)]
    query_prefix: Option<String>,

    #[arg(long, global = true)]
    passage_prefix: Option<String>,

    #[command(subcommand)]
    command: Option<TopCommand>,
}

#[derive(Subcommand)]
enum TopCommand {
    #[command(about = "Run the daemon process")]
    Daemon {
        #[arg(short, long)]
        model: String,
        #[arg(long)]
        query_prefix: Option<String>,
        #[arg(long)]
        passage_prefix: Option<String>,
    },
    #[command(about = "DSL graph dump for debugging")]
    Dsl {
        #[command(subcommand)]
        command: dsl::DslCommands,
        #[arg(short, long, help = "Project root path")]
        root: std::path::PathBuf,
    },
    #[command(flatten)]
    Cli(cli::CliCommand),
}

fn init_tracing(default_level: &str) {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(default_level)),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
}

fn main() -> Result<()> {
    let top = TopLevel::parse();

    if top.mcp {
        init_tracing("warn");
        return mcp::run_mcp();
    }

    let model = top.model.as_deref();
    let query_prefix = top.query_prefix.as_deref();
    let passage_prefix = top.passage_prefix.as_deref();

    match top.command {
        None => {
            init_tracing("warn");
            tui::run_tui(model, query_prefix, passage_prefix)
        }
        Some(TopCommand::Daemon {
            model,
            query_prefix,
            passage_prefix,
        }) => {
            let config = zti_daemon::DaemonConfig {
                model: Cow::Owned(model),
                query_prefix: query_prefix.as_deref(),
                passage_prefix: passage_prefix.as_deref(),
            };
            zti_daemon::run_daemon(&config)
        }
        Some(TopCommand::Dsl { command, root }) => {
            init_tracing("warn");
            dsl::run_dsl(&root, command)
        }
        Some(TopCommand::Cli(cmd)) => {
            init_tracing("warn");
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cli::run(cmd, model, query_prefix, passage_prefix))
        }
    }
}
