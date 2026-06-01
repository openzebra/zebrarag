use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::Subcommand;

use zti_dsl::chunking::chunk_text_file;
use zti_dsl::render::dsl::{DslRenderer, render_files_only};
use zti_dsl::render::tree::AsciiTreeRenderer;
use zti_dsl::DslChunker;
use zti_tree_sitter::{frontend_for, parse_kinds, parse_language};
use zti_ts_core::walker::LanguageFrontend;

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
    #[command(about = "Look up a symbol by name: kind, location, doc, callers/callees, body")]
    SearchDep {
        #[arg(short, long, help = "Symbol/type/function name (bare or qualified)")]
        name: String,
        #[arg(short, long, help = "Dependency/crate name to search in (resolves path from cargo/pub/npm caches)")]
        lib: Option<String>,
        #[arg(long, default_value = "2", help = "Call-graph depth")]
        depth: usize,
    },
    #[command(about = "Sequential chunk trace to diagnose chunk-generation hangs")]
    ChunkTrace,
}

fn dep_version_from_lock(root: &Path, lib: &str) -> Option<String> {
    let lock = root.join("Cargo.lock");
    if lock.is_file() {
        let content = std::fs::read_to_string(lock).ok()?;
        if let Ok(toml) = content.parse::<toml::Value>()
            && let Some(packages) = toml.get("package").and_then(|v| v.as_array())
        {
            for pkg in packages {
                if pkg.get("name").and_then(|v| v.as_str()) == Some(lib) {
                    return pkg.get("version").and_then(|v| v.as_str()).map(String::from);
                }
            }
        }
    }

    let pkg = root.join("package.json");
    if pkg.is_file()
        && let Ok(content) = std::fs::read_to_string(&pkg)
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&content)
    {
        if let Some(deps) = json.get("dependencies").and_then(|d| d.get(lib)) {
            return deps.as_str().map(|s| s.trim_start_matches('^').to_string());
        }
        if let Some(deps) = json.get("devDependencies").and_then(|d| d.get(lib)) {
            return deps.as_str().map(|s| s.trim_start_matches('^').to_string());
        }
    }

    None
}

fn resolve_dep_path(root: &Path, lib: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok();
    let want_version = dep_version_from_lock(root, lib);

    // Project-local deps (JS/TS/Solidity npm, Foundry git submodules)
    let locals = [
        root.join("node_modules").join(lib),
        root.join("lib").join(lib),
    ];
    for p in locals {
        if p.is_dir() {
            return Some(p);
        }
    }

    // Cargo registry: ~/.cargo/registry/src/<hash>/<crate>-<version>/
    if let Some(ref h) = home {
        let cargo_dir = Path::new(h).join(".cargo/registry/src");
        if let Ok(entries) = std::fs::read_dir(&cargo_dir) {
            let prefix = format!("{lib}-");
            let mut best: Option<PathBuf> = None;
            for entry in entries.flatten() {
                if let Ok(sub) = std::fs::read_dir(entry.path()) {
                    for dep in sub.flatten() {
                        let fname_os = dep.file_name();
                        let fname = fname_os.to_string_lossy();
                        if !fname.starts_with(&prefix) || !dep.path().is_dir() {
                            continue;
                        }
                        if let Some(ref ver) = want_version {
                            let expected = format!("{lib}-{ver}");
                            if fname.as_ref() == expected {
                                return Some(dep.path());
                            }
                        }
                        if best.as_ref().is_none_or(|b| dep.path() > *b) {
                            best = Some(dep.path());
                        }
                    }
                }
            }
            if let Some(p) = best {
                return Some(p);
            }
        }
    }

    // pub.dev: ~/.pub-cache/hosted/pub.dev/<name>-*/
    if let Some(ref h) = home {
        let pub_dir = Path::new(h).join(".pub-cache/hosted/pub.dev");
        if let Ok(entries) = std::fs::read_dir(&pub_dir) {
            for entry in entries.flatten() {
                let name_os = entry.file_name();
                let name = name_os.to_string_lossy();
                if name.starts_with(lib) && entry.path().is_dir() {
                    return Some(entry.path());
                }
            }
        }
    }

    None
}

pub fn run_dsl(root: &Path, command: DslCommands) -> Result<()> {
    // Handle SearchDep with --lib early: resolve dep path, build its index, search.
    if let DslCommands::SearchDep {
        name: ref name_lib,
        lib: Some(ref lib_name),
        depth,
    } = command
    {
        let dep_root = resolve_dep_path(root, lib_name).ok_or_else(|| {
            anyhow::anyhow!("dependency '{lib_name}' not found in cargo registry, npm, or pub cache")
        })?;
        let (index, _text_files) = zti_dsl::build_index(dep_root.to_string_lossy().as_ref())?;
        match zti_dsl::resolve_name(&index, name_lib) {
            zti_dsl::NameMatch::Found(id) => {
                print!("{}", zti_dsl::render_symbol_overview(&index, id, depth, 6000))
            }
            zti_dsl::NameMatch::Ambiguous(ref ids) => {
                print!("{}", zti_dsl::search_dep::render_candidates(&index, ids))
            }
            zti_dsl::NameMatch::NotFound => {
                return Err(anyhow::anyhow!("no symbol '{name_lib}' in '{lib_name}'"))
            }
        }
        return Ok(());
    }

    let canonical = root.canonicalize()?;
    let root_cow = canonical.to_string_lossy();

    let (index, text_files) = zti_dsl::build_index(&root_cow)?;
    tracing::info!(
        "{} symbols, {} edges, {} files (code), {} text files",
        index.symbols.len(),
        index.edges.len(),
        index.files.len(),
        text_files.len(),
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
        DslCommands::SearchDep { ref name, depth, .. } => {
            match zti_dsl::resolve_name(&index, name) {
                zti_dsl::NameMatch::Found(id) => {
                    print!("{}", zti_dsl::render_symbol_overview(&index, id, depth, 6000))
                }
                zti_dsl::NameMatch::Ambiguous(ref ids) => {
                    print!("{}", zti_dsl::search_dep::render_candidates(&index, ids))
                }
                zti_dsl::NameMatch::NotFound => {
                    return Err(anyhow::anyhow!("no symbol named '{name}'"))
                }
            }
        }
        DslCommands::ChunkTrace => {
            const CHARS_PER_TOKEN: usize = 4;

            let chunker = DslChunker::new(&index);

            let mut terminal_cache: HashMap<zti_tree_sitter::Language, Vec<u16>> =
                HashMap::with_capacity(4);
            for lang in index.files.iter().map(|f| f.language) {
                if terminal_cache.contains_key(&lang) {
                    continue;
                }
                let frontend = frontend_for(lang);
                let ts_lang = frontend.language();
                let names = frontend.config().terminal_node_kinds;
                let mut ids = Vec::with_capacity(names.len());
                for name in names {
                    let id = ts_lang.id_for_node_kind(name, true);
                    if id != 0 {
                        ids.push(id);
                    }
                }
                terminal_cache.insert(lang, ids);
            }

            let code_total = index.files.len();
            let text_total = text_files.len();
            eprintln!(
                "terminal_cache: {} languages, {} code files, {} text files",
                terminal_cache.len(),
                code_total,
                text_total,
            );

            let sizing = zti_recursive_chunk::ChunkConfig {
                chunk_size: 2048,
                min_chunk_size: 512,
                chunk_overlap: 200,
            };

            let total = code_total + text_total;
            let mut total_chunks = 0usize;
            let mut total_sub = 0usize;
            let mut total_est_tokens = 0usize;
            let trace_start = Instant::now();

            for (i, file) in index.files.iter().enumerate() {
                let contents = match std::fs::read_to_string(&file.path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "DEBUG [{}/{}] {} - skip (read error: {})",
                            i + 1,
                            total,
                            file.path,
                            e
                        );
                        continue;
                    }
                };

                let f_start = Instant::now();
                let chunks = chunker.chunks_for_file(&file.path, &contents);
                let f_locate = f_start.elapsed();

                let file_tokens = contents.len() / CHARS_PER_TOKEN;
                total_est_tokens += file_tokens;
                eprintln!(
                    "DEBUG [{}/{}] {} ({}B, ~{} tok, {}) -> {} chunks in {:?}{}",
                    i + 1,
                    total,
                    file.path,
                    contents.len(),
                    file_tokens,
                    file.language.as_str(),
                    chunks.len(),
                    f_locate,
                    if f_locate.as_millis() > 500 { " WARN" } else { "" },
                );

                let frontend = frontend_for(file.language);
                let ts_lang = frontend.language();
                let terminal_ids = terminal_cache
                    .get(&file.language)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                for (ci, chunk) in chunks.iter().enumerate() {
                    let c_start = Instant::now();
                    let sub = zti_recursive_chunk::split_text(
                        &chunk.body,
                        &sizing,
                        Some(&ts_lang),
                        terminal_ids,
                    );
                    let c_elapsed = c_start.elapsed();

                    if c_elapsed.as_millis() > 50 {
                        eprintln!(
                            "DEBUG   [{}/{}] sym={} kind={:?} body={}B -> {} sub in {:?}{}",
                            ci + 1,
                            chunks.len(),
                            chunk.sym_id,
                            chunk.kind,
                            chunk.body.len(),
                            sub.len(),
                            c_elapsed,
                            if c_elapsed.as_millis() > 500 { " WARN" } else { "" },
                        );
                    }

                    total_sub += sub.len();
                }

                total_chunks += chunks.len();

                let f_total = f_start.elapsed();
                if f_total.as_millis() > 1000 {
                    eprintln!("DEBUG   WARN: file took {:?}", f_total);
                }

                let _ = std::io::stderr().flush();
            }

            // Process text files — move ownership out with into_iter (no clone).
            for (ti, (path, contents)) in text_files.into_iter().enumerate() {
                let i = code_total + ti;
                let bytes = contents.len();
                let est_tokens = bytes / CHARS_PER_TOKEN;
                total_est_tokens += est_tokens;
                let f_start = Instant::now();

                let rel = path
                    .strip_prefix(root_cow.as_ref())
                    .unwrap_or(&path)
                    .trim_start_matches('/')
                    .to_string();
                let chunk = chunk_text_file(rel, path, contents);
                let f_locate = f_start.elapsed();

                eprintln!(
                    "DEBUG [{}/{}] {} ({}B, ~{} tok, text) -> 1 chunk in {:?}{}",
                    i + 1,
                    total,
                    chunk.file,
                    bytes,
                    est_tokens,
                    f_locate,
                    if f_locate.as_millis() > 500 { " WARN" } else { "" },
                );

                let c_start = Instant::now();
                let sub = zti_recursive_chunk::split_text(
                    &chunk.body,
                    &sizing,
                    None,
                    &[],
                );
                let c_elapsed = c_start.elapsed();

                if c_elapsed.as_millis() > 50 {
                    eprintln!(
                        "DEBUG   [1/1] sym=N/A kind=Document body={}B -> {} sub in {:?}{}",
                        chunk.body.len(),
                        sub.len(),
                        c_elapsed,
                        if c_elapsed.as_millis() > 500 { " WARN" } else { "" },
                    );
                }

                total_sub += sub.len();
                total_chunks += 1;

                let f_total = f_start.elapsed();
                if f_total.as_millis() > 1000 {
                    eprintln!("DEBUG   WARN: file took {:?}", f_total);
                }

                let _ = std::io::stderr().flush();
            }

            let elapsed = trace_start.elapsed();
            println!();
            println!("--- Chunk Trace Summary ---");
            println!(
                "Files: {total} (code={code_total}, text={text_total})",
            );
            println!(
                "Chunks: {total_chunks}, Sub-chunks: {total_sub}",
            );
            println!(
                "Bytes: ~{} MB, Est. tokens: ~{total_est_tokens}, Time: {:.2}s",
                total_est_tokens * CHARS_PER_TOKEN / (1024 * 1024),
                elapsed.as_secs_f64(),
            );
        }
    }

    Ok(())
}
