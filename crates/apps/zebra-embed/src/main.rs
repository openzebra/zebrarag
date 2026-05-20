use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use tracing_subscriber::EnvFilter;

use zti_embed::OnnxVariant;
use zti_ipc_client::Client;
use zti_protocol::format_search_results;
use zti_protocol::request::*;
use zti_protocol::response::*;

#[derive(Parser)]
#[command(
    name = "zebra-embed",
    version,
    about = "Index / search / chat via daemon"
)]
struct Cli {
    #[arg(short, long, global = true)]
    model: Option<String>,

    #[arg(long, value_enum, global = true)]
    variant: Option<OnnxVariant>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "Index a project")]
    Index {
        #[arg(short, long)]
        root: PathBuf,
        #[arg(long)]
        refresh: bool,
    },
    #[command(about = "Search a project")]
    Search {
        #[arg(short, long)]
        root: PathBuf,
        query: String,
        #[arg(short, long, default_value = "5")]
        limit: usize,
        #[arg(long)]
        lang: Option<String>,
        #[arg(long)]
        glob: Option<String>,
        #[arg(long, default_value = "false")]
        exhaustive: bool,
    },
    #[command(about = "Interactive chat (search loop)")]
    Chat {
        #[arg(short, long)]
        root: PathBuf,
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    #[command(about = "Show project status")]
    Status {
        #[arg(short, long)]
        root: Option<PathBuf>,
    },
    #[command(about = "Run diagnostics")]
    Doctor {
        #[arg(short, long)]
        root: Option<PathBuf>,
    },
    #[command(about = "Show daemon environment")]
    Env,
    #[command(about = "Stop the daemon")]
    Stop,
    #[command(about = "Remove a project")]
    Remove {
        #[arg(short, long)]
        root: PathBuf,
    },
}

/// Connect to the daemon (auto-spawning if needed) and complete the mandatory
/// handshake. Every subcommand uses this — including `Stop`. Replaces what was
/// previously ~10 lines of duplicated boilerplate per command.
async fn open_client(model: Option<&str>, variant: Option<&str>) -> Result<Client> {
    let mut client = Client::connect(Duration::from_secs(10), model, variant).await?;
    client.handshake().await?;
    Ok(client)
}

fn canon(p: &Path) -> Result<String> {
    Ok(p.canonicalize()?.to_string_lossy().into_owned())
}

fn canon_opt(p: Option<PathBuf>) -> Result<Option<String>> {
    p.map(|r| canon(&r)).transpose()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    let model = cli.model.as_deref();
    let variant: Option<&'static str> = cli.variant.and_then(|v| match v {
        OnnxVariant::Auto => None,
        other => Some(other.as_str()),
    });
    match cli.command {
        Commands::Index { root, refresh } => {
            let mut client = open_client(model, variant).await?;
            let project_root = canon(&root)?;
            let bar = RefCell::new(None::<ProgressBar>);

            let resp = client
                .request_streaming(
                    Request::Index(IndexReq { project_root, refresh }),
                    |frame| {
                        if let Response::IndexProgress(p) = frame {
                            let mut slot = bar.borrow_mut();
                            match p.phase.as_str() {
                                "start" => {
                                    if let Some(old) = slot.take() {
                                        old.finish_and_clear();
                                    }
                                    let b = ProgressBar::new(p.total);
                                    b.set_style(
                                        ProgressStyle::with_template(
                                            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                                        )
                                        .unwrap_or_else(|_| ProgressStyle::default_bar()),
                                    );
                                    *slot = Some(b);
                                }
                                "embed" => {
                                    if let Some(b) = slot.as_ref() {
                                        b.set_position(p.current);
                                    }
                                }
                                "finish" => {
                                    if let Some(b) = slot.take() {
                                        b.finish_with_message(p.message);
                                    }
                                }
                                _ => {}
                            }
                        }
                    },
                )
                .await?;

            match resp {
                Response::Index(Ok(stats)) => {
                    println!(
                        "Indexed {} chunks in {} files ({:.1}s)",
                        stats.total_chunks,
                        stats.total_files,
                        stats.duration_ms as f64 / 1000.0
                    );
                }
                Response::Index(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Search {
            root,
            query,
            limit,
            lang,
            glob,
            exhaustive,
        } => {
            let mut client = open_client(model, variant).await?;
            let project_root = canon(&root)?;
            let resp = client
                .request(Request::Search(SearchReq {
                    project_root,
                    query,
                    limit,
                    offset: None,
                    languages: lang.map(|l| l.split(',').map(String::from).collect()),
                    path_glob: glob,
                    refresh_index: false,
                    exhaustive,
                }))
                .await?;
            match resp {
                Response::Search(Ok(results)) => {
                    print!("{}", format_search_results(&results));
                    println!("{} results", results.total);
                }
                Response::Search(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Chat { root, limit } => {
            let mut client = open_client(model, variant).await?;
            let project_root = canon(&root)?;
            let mut rl = rustyline::DefaultEditor::new()?;
            println!("zebra-embed chat — type a query, :q or Ctrl-D to exit.");

            while let Ok(line) = rl.readline("> ") {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == ":q" {
                    break;
                }
                let _ = rl.add_history_entry(trimmed);

                let resp = client
                    .request(Request::Search(SearchReq {
                        project_root: project_root.clone(),
                        query: trimmed.to_string(),
                        limit,
                        offset: None,
                        languages: None,
                        path_glob: None,
                        refresh_index: false,
                        exhaustive: false,
                    }))
                    .await?;
                match resp {
                    Response::Search(Ok(results)) => {
                        print!("{}", format_search_results(&results));
                    }
                    Response::Search(Err(e)) => eprintln!("Error: {}", e.message),
                    other => eprintln!("Unexpected response: {:?}", other),
                }
            }
        }
        Commands::Status { root } => {
            let mut client = open_client(model, variant).await?;
            let project_root = canon_opt(root)?;
            let resp = client
                .request(Request::ProjectStatus(ProjectStatusReq { project_root }))
                .await?;
            match resp {
                Response::ProjectStatus(Ok(status)) => {
                    println!("Root: {}", status.project_root);
                    println!("Model: {} (dim={})", status.model_id, status.model_dim);
                    println!("Chunks: {}", status.total_chunks);
                    println!("Files: {}", status.total_files);
                }
                Response::ProjectStatus(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Doctor { root } => {
            let mut client = open_client(model, variant).await?;
            let project_root = canon_opt(root)?;
            let resp = client
                .request(Request::Doctor(DoctorReq { project_root }))
                .await?;
            match resp {
                Response::Doctor(Ok(report)) => {
                    println!("Device: {}", report.device);
                    for check in &report.checks {
                        let marker = match check.status {
                            CheckStatus::Ok => "OK  ",
                            CheckStatus::Warn => "WARN",
                            CheckStatus::Err => "ERR ",
                        };
                        println!("[{}] {}: {}", marker, check.name, check.message);
                    }
                }
                Response::Doctor(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Env => {
            let mut client = open_client(model, variant).await?;
            let resp = client.request(Request::DaemonEnv).await?;
            match resp {
                Response::DaemonEnv(env) => {
                    println!("Data dir: {}", env.data_dir);
                    println!("Socket: {}", env.socket_path);
                    println!("Model: {}", env.model_id);
                    println!("Device: {}", env.device);
                    println!("CPUs: {}", env.cpus);
                    println!("RAM: {} MB", env.mem_total_mb);
                }
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Stop => {
            let mut client = open_client(model, variant).await?;
            let resp = client.request(Request::Stop).await?;
            if matches!(resp, Response::Stop(())) {
                println!("Daemon stopped.");
            }
        }
        Commands::Remove { root } => {
            let mut client = open_client(model, variant).await?;
            let project_root = canon(&root)?;
            let resp = client
                .request(Request::RemoveProject(RemoveProjectReq { project_root }))
                .await?;
            match resp {
                Response::RemoveProject(Ok(())) => println!("Project removed."),
                Response::RemoveProject(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
    }

    Ok(())
}
