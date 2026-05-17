use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use zti_ipc_client::Client;
use zti_protocol::request::*;
use zti_protocol::response::*;

#[derive(Parser)]
#[command(name = "zebra-embed", version, about = "Index / search / chat via daemon")]
struct Cli {
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
        #[arg(short, long)]
        lang: Option<String>,
        #[arg(short, long)]
        glob: Option<String>,
    },
    #[command(about = "Interactive chat")]
    Chat {
        #[arg(short, long)]
        root: PathBuf,
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Index { root, refresh } => {
            let mut client = Client::connect(Duration::from_secs(10)).await?;
            client.handshake().await?;

            let root_str = root.canonicalize()?.to_string_lossy().to_string();

            use indicatif::{ProgressBar, ProgressStyle};
            use std::cell::RefCell;
            let bar = RefCell::new(None::<ProgressBar>);

            let resp = client
                .request_streaming(
                    Request::Index(IndexReq {
                        project_root: root_str,
                        refresh,
                    }),
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
                Response::Index(Err(e)) => {
                    eprintln!("Error: {}", e.message);
                }
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Search { root, query, limit, lang, glob } => {
            let mut client = Client::connect(Duration::from_secs(10)).await?;
            client.handshake().await?;

            let root_str = root.canonicalize()?.to_string_lossy().to_string();
            let resp = client.request(Request::Search(SearchReq {
                project_root: root_str,
                query,
                limit,
                offset: None,
                languages: lang.map(|l| l.split(',').map(String::from).collect()),
                path_glob: glob,
                refresh_index: false,
            })).await?;

            match resp {
                Response::Search(Ok(results)) => {
                    for (i, hit) in results.hits.iter().enumerate() {
                        println!("#{} {:.4} {} ({}:{}-{})", i + 1, hit.score, hit.symbol_qualified,
                            hit.file_path, hit.start_line, hit.end_line);
                    }
                    println!("{} results", results.total);
                }
                Response::Search(Err(e)) => {
                    eprintln!("Error: {}", e.message);
                }
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Chat { .. } => {
            eprintln!("chat mode requires a running daemon with indexed project");
        }
        Commands::Status { root } => {
            let mut client = Client::connect(Duration::from_secs(10)).await?;
            client.handshake().await?;

            let root_str = root.map(|r| r.canonicalize().map(|p| p.to_string_lossy().to_string()))
                .transpose()?;
            let resp = client.request(Request::ProjectStatus(ProjectStatusReq {
                project_root: root_str,
            })).await?;

            match resp {
                Response::ProjectStatus(Ok(status)) => {
                    println!("Root: {}", status.project_root);
                    println!("Model: {} (dim={})", status.model_id, status.model_dim);
                }
                Response::ProjectStatus(Err(e)) => {
                    eprintln!("Error: {}", e.message);
                }
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Doctor { root } => {
            let mut client = Client::connect(Duration::from_secs(10)).await?;
            client.handshake().await?;

            let resp = client.request(Request::Doctor(DoctorReq {
                project_root: root.map(|r| r.canonicalize().map(|p| p.to_string_lossy().to_string()))
                    .transpose()?,
            })).await?;

            match resp {
                Response::Doctor(Ok(report)) => {
                    println!("Model: {} ({})", report.model_path, if report.model_ok { "OK" } else { "MISSING" });
                    println!("DB: {} ({})", report.db_path, if report.db_ok { "OK" } else { "MISSING" });
                    println!("Device: {}", report.device);
                }
                Response::Doctor(Err(e)) => {
                    eprintln!("Error: {}", e.message);
                }
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        Commands::Env => {
            let mut client = Client::connect(Duration::from_secs(10)).await?;
            client.handshake().await?;

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
            let mut client = Client::connect(Duration::from_secs(5)).await?;
            let resp = client.request(Request::Stop).await?;
            if matches!(resp, Response::Stop(())) {
                println!("Daemon stopped.");
            }
        }
        Commands::Remove { root } => {
            let mut client = Client::connect(Duration::from_secs(10)).await?;
            client.handshake().await?;

            let root_str = root.canonicalize()?.to_string_lossy().to_string();
            let resp = client.request(Request::RemoveProject(RemoveProjectReq {
                project_root: root_str,
            })).await?;

            match resp {
                Response::RemoveProject(Ok(())) => println!("Project removed."),
                Response::RemoveProject(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
    }

    Ok(())
}
