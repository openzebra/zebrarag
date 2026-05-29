use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use clap::Subcommand;
use indicatif::{ProgressBar, ProgressStyle};

use zti_ipc_client::Client;
use zti_protocol::format_search_results;
use zti_protocol::request::*;
use zti_protocol::response::*;

#[derive(Subcommand)]
pub enum CliCommand {
    #[command(about = "Index a project")]
    Index {
        #[arg(short, long, help = "Project name, index number, or root path")]
        root: PathBuf,
        #[arg(long)]
        refresh: bool,
    },
    #[command(about = "Search a project")]
    Search {
        #[arg(short, long, help = "Project name, index number, or root path")]
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
        #[arg(long, default_value_t = SearchMode::Query)]
        mode: SearchMode,
    },
    #[command(about = "Interactive chat (search loop)")]
    Chat {
        #[arg(short, long, help = "Project name, index number, or root path")]
        root: PathBuf,
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    #[command(about = "Show project status")]
    Status {
        #[arg(short, long, help = "Project name, index number, or root path")]
        root: Option<PathBuf>,
    },
    #[command(about = "Run diagnostics")]
    Doctor {
        #[arg(short, long, help = "Project name, index number, or root path")]
        root: Option<PathBuf>,
    },
    #[command(about = "Show daemon environment")]
    Env,
    #[command(about = "Stop the daemon")]
    Stop,
    #[command(about = "Remove a project")]
    Remove {
        #[arg(short, long, help = "Project name, index number, or root path")]
        root: PathBuf,
    },
    #[command(about = "List all indexed projects")]
    Projects,
}

async fn open_client(
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
    model_dtype: Option<&str>,
) -> Result<Client> {
    let mut client = Client::connect(
        Duration::from_secs(60),
        model,
        query_prefix,
        passage_prefix,
        model_dtype,
    )
    .await?;
    client.handshake().await?;
    Ok(client)
}

async fn resolve_root(root: &Path) -> Result<String> {
    zti_store::resolve_project(Some(&root.to_string_lossy())).await
}

pub async fn run(
    cmd: CliCommand,
    model: Option<&str>,
    query_prefix: Option<&str>,
    passage_prefix: Option<&str>,
    model_dtype: Option<&str>,
) -> Result<()> {
    let open = || open_client(model, query_prefix, passage_prefix, model_dtype);

    match cmd {
        CliCommand::Index { root, refresh } => {
            let mut client = open().await?;
            let project_root = match resolve_root(&root).await {
                Ok(p) => p,
                Err(_) => root
                    .canonicalize()
                    .map_err(|_| anyhow::anyhow!("Project directory not found: {}", root.display()))?
                    .to_string_lossy()
                    .to_string(),
            };
            let bar = RefCell::new(None::<ProgressBar>);

            let resp = client
                .request_streaming(
                    Request::Index(IndexReq {
                        project_root,
                        refresh,
                        search_method: None,
                    }),
                    |frame| {
                        if let Response::IndexProgress(p) = frame {
                            let mut slot = bar.borrow_mut();
                            match p.phase {
                                zti_protocol::response::IndexPhase::Start => {
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
                                zti_protocol::response::IndexPhase::Gather
                                | zti_protocol::response::IndexPhase::Tokenize => {
                                    if let Some(b) = slot.as_ref() {
                                        b.set_position(p.current);
                                        b.set_message(p.message);
                                    }
                                }
                                zti_protocol::response::IndexPhase::Embed => {
                                    if let Some(b) = slot.as_ref() {
                                        b.set_position(p.current);
                                    }
                                }
                                zti_protocol::response::IndexPhase::BuildIndex => {
                                    if let Some(b) = slot.as_ref() {
                                        b.set_message(p.message);
                                    }
                                }
                                zti_protocol::response::IndexPhase::Finish => {
                                    if let Some(b) = slot.take() {
                                        b.finish_with_message(p.message);
                                    }
                                }
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
        CliCommand::Search {
            root,
            query,
            limit,
            lang,
            glob,
            exhaustive,
            mode,
        } => {
            let mut client = open().await?;
            let project_root = resolve_root(&root).await?;
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
                    mode,
                }))
                .await?;
            match resp {
                Response::Search(Ok(results)) => {
                    print!("{}", format_search_results(&results));
                    if results.total == 0 {
                        println!("0 results (hint: project may not be indexed; run `zebraindex index --root <root>` to index)");
                    } else {
                        println!("{} results", results.total);
                    }
                }
                Response::Search(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        CliCommand::Chat { root, limit } => {
            let mut client = open().await?;
            let project_root = resolve_root(&root).await?;
            let mut rl = rustyline::DefaultEditor::new()?;
            println!("zebraindex chat — type a query, :q or Ctrl-D to exit.");

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
                        mode: SearchMode::default(),
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
        CliCommand::Status { root } => {
            let mut client = open().await?;
            let project_root = match root {
                Some(r) => Some(resolve_root(&r).await?),
                None => None,
            };
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
        CliCommand::Doctor { root } => {
            let mut client = open().await?;
            let project_root = match root {
                Some(r) => Some(resolve_root(&r).await?),
                None => None,
            };
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
        CliCommand::Env => {
            let mut client = open().await?;
            let resp = client.request(Request::DaemonEnv).await?;
            match resp {
                Response::DaemonEnv(env) => {
                    println!("Data dir: {}", env.data_dir);
                    println!("Socket: {}", env.socket_path);
                    println!("Model: {}", env.model_id);
                    println!("Device: {}", env.device);
                    println!("CPUs: {}", env.cpus);
                    println!("RAM: {} MB", env.mem_total_mb);
                    if let Some(ref p) = env.query_prefix {
                        println!("Query prefix: {:?}", p);
                    } else {
                        println!("Query prefix: None");
                    }
                    if let Some(ref p) = env.passage_prefix {
                        println!("Passage prefix: {:?}", p);
                    } else {
                        println!("Passage prefix: None");
                    }
                }
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        CliCommand::Stop => {
            let mut client = open().await?;
            let resp = client.request(Request::Stop).await?;
            if matches!(resp, Response::Stop(())) {
                println!("Daemon stopped.");
            }
        }
        CliCommand::Remove { root } => {
            let mut client = open().await?;
            let project_root = resolve_root(&root).await?;
            let resp = client
                .request(Request::RemoveProject(RemoveProjectReq { project_root }))
                .await?;
            match resp {
                Response::RemoveProject(Ok(())) => println!("Project removed."),
                Response::RemoveProject(Err(e)) => eprintln!("Error: {}", e.message),
                other => eprintln!("Unexpected response: {:?}", other),
            }
        }
        CliCommand::Projects => {
            let projects = zti_store::list_projects().await?;
            if projects.is_empty() {
                println!("No indexed projects found.");
                return Ok(());
            }
            println!("| # | Project | Root | Model | Chunks | Files | Last Indexed |");
            println!("|---|---------|------|-------|--------|-------|-------------|");
            for (i, p) in projects.iter().enumerate() {
                let name = Path::new(&p.root_path)
                    .file_name()
                    .map(|s| s.to_string_lossy())
                    .unwrap_or_else(|| std::borrow::Cow::Borrowed(&p.root_path));
                let ago = zti_common::format::format_elapsed(p.last_indexed_ns);
                println!(
                    "| {} | {} | {} | {} | {} | {} | {} |",
                    i + 1,
                    name,
                    p.root_path,
                    p.model_id,
                    p.total_chunks,
                    p.total_files,
                    ago
                );
            }
            println!("\n{} project(s)", projects.len());
        }
    }

    Ok(())
}
