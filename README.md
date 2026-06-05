<div align="center">

# 🦓 ZebraIndex

### Semantic Code Intelligence for AI Coding Agents

**On-device embedding · AST-aware chunking · Incremental indexing · MCP-native**

### [Documentation & Website →](https://zebra.sh)

[![github](https://img.shields.io/badge/github-hicaru/zebra__tree__indexer-181717?style=flat-square&logo=github)](https://github.com/hicaru/zebra_tree_indexer)
[![rust](https://img.shields.io/badge/rust-1.91+-db6d28?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-5B5BD6?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![telegram](https://img.shields.io/badge/telegram-join-26A5E4?style=flat-square&logo=telegram)](https://t.me/+MLRSdmyS6CM4YzAy)

[![Windows](https://img.shields.io/badge/Windows-supported-blue.svg)](#supported-platforms)
[![macOS](https://img.shields.io/badge/macOS-supported-blue.svg)](#supported-platforms)
[![Linux](https://img.shields.io/badge/Linux-supported-blue.svg)](#supported-platforms)
[![FreeBSD](https://img.shields.io/badge/FreeBSD-supported-blue.svg)](#supported-platforms)
[![OpenBSD](https://img.shields.io/badge/OpenBSD-supported-blue.svg)](#supported-platforms)

[![Claude Code](https://img.shields.io/badge/Claude_Code-supported-blueviolet.svg)](#supported-agents)
[![Cursor](https://img.shields.io/badge/Cursor-supported-blueviolet.svg)](#supported-agents)
[![Codex](https://img.shields.io/badge/Codex-supported-blueviolet.svg)](#supported-agents)
[![opencode](https://img.shields.io/badge/opencode-supported-blueviolet.svg)](#supported-agents)
[![Hermes Agent](https://img.shields.io/badge/Hermes_Agent-supported-blueviolet.svg)](#supported-agents)
[![Gemini](https://img.shields.io/badge/Gemini-supported-blueviolet.svg)](#supported-agents)
[![Antigravity](https://img.shields.io/badge/Antigravity-supported-blueviolet.svg)](#supported-agents)
[![Kiro](https://img.shields.io/badge/Kiro-supported-blueviolet.svg)](#supported-agents)
[![Pi](https://img.shields.io/badge/Pi-supported-blueviolet.svg)](#supported-agents)

[![CUDA](https://img.shields.io/badge/CUDA-accelerated-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Metal](https://img.shields.io/badge/Metal-accelerated-8a8a8a?style=flat-square&logo=apple)](https://developer.apple.com/metal/)
[![CPU](https://img.shields.io/badge/CPU-fallback-ff6f00?style=flat-square)]()

</div>

---

## What is Zebra?

Zebra indexes your codebase **locally** — parsing source files with [tree-sitter](https://tree-sitter.github.io/), embedding each symbol and chunk with an **on-device transformer model** (via [Candle](https://github.com/huggingface/candle)), and storing everything in [LanceDB](https://lancedb.com/) — so AI agents can **search code by meaning, not by string matching.**

It runs entirely on your machine. **No data leaves your computer. No API keys. No cloud dependency.**

When an agent explores a codebase, it spawns sub-agents that scan files with grep, glob, and Read — consuming tokens on every tool call. **Zebra gives those agents a pre-built semantic index** — symbol relationships, call graphs, and code structure — so they query the index instantly instead of scanning files.

---

## Get Started

### 1. Install via Cargo

```bash
cargo install --git https://github.com/hicaru/zebra_tree_indexer
```

Or build from source:

```bash
git clone https://github.com/hicaru/zebra_tree_indexer.git
cd zebra_tree_indexer
cargo build --release -p zebraindex
```

### 2. Launch the TUI

The terminal UI walks you through model selection, downloads, daemon launch, and indexing:

```bash
zebraindex
```

### 3. Index & Search

```bash
# Index a project
zebraindex index -r /path/to/your/project

# Search by meaning
zebraindex search -r /path/to/your/project "rate limiting middleware"

# Interactive search loop
zebraindex chat -r /path/to/your/project
```

### 4. Wire up your agent(s)

Run as an MCP server so your agent can search your codebase semantically:

```bash
zebraindex --mcp
```

Add to **Claude Code** (`~/.claude.json`):

```json
{
  "mcpServers": {
    "zebra-mcp": {
      "command": "zebraindex",
      "args": ["--mcp"]
    }
  }
}
```

Or via CLI:

```bash
claude mcp add -s user zebra-mcp -- zebraindex --mcp
```

Add to **Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "zebra-mcp": {
      "command": "zebraindex",
      "args": ["--mcp"]
    }
  }
}
```

Add to **opencode** (`~/.config/opencode/opencode.json`):

```json
"zebra-mcp": {
  "type": "local",
  "command": ["zebraindex", "--mcp"],
  "enabled": true
}
```

Add to **Codex CLI** (`~/.codex/config.toml`):

```toml
[mcp_servers.zebra-mcp]
command = "zebraindex"
args = ["--mcp"]
```

Add to **Pi** (`.pi/config.toml`):

```toml
[mcp_servers.zebra-mcp]
command = "zebraindex"
args = ["--mcp"]
```

Add to **Gemini CLI**, **Hermes Agent**, **Antigravity IDE**, or **Kiro** — same pattern: point the MCP server config to `zebraindex --mcp`.

<sub>Zebra is MCP-native — works with **any** MCP-compatible agent. The agent gets seven tools: `searchQuery`, `searchPassage`, `searchDep`, `fileTree`, `projectList`, `doctor`, and `projectList`.</sub>

---

## Key Features

| | |
|---|---|
| **🔍 Semantic Search** | Search code by intent, not by string. `"user session validation"` finds the session validation logic, not files containing that literal string. |
| **📋 Search by Example** | Paste a code snippet or error message — find semantically similar implementations across the codebase. |
| **🧠 Symbol Lookup** | Look up any symbol by name and get its definition with full call graph (callers & callees), doc summary, and source body — all in one call. |
| **⚡ Incremental Indexing** | Tracks file snapshots. Only re-indexes what changed. Δ-only — no full rebuilds. |
| **🔒 100% Local** | No data leaves your machine. No API keys. No external services. LanceDB + Candle run on-device. |
| **🌳 AST-Aware Chunking** | tree-sitter parses every source file. Chunks are semantic units (functions, structs, methods) — not arbitrary line splits. |
| **🔄 Auto-Detect Models** | Downloads embedding models from HuggingFace automatically. No manual config. |
| **🦀 Rust Native** | Single binary. Zero-cost abstractions. Parallel chunking with rayon. Async daemon with tokio. |
| **🔌 MCP-Native** | First-class MCP server via rmcp. Works with Claude Code, Cursor, Codex, opencode, Hermes Agent, Gemini, Antigravity, Kiro, Pi, and any MCP client. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AI Agent                                     │
│                                                                     │
│   "How does request validation work in this codebase?"              │
│       calls zebra-mcp tools directly — no grep/find needed          │
│                                 │                                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │  MCP (stdio)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      zebraindex (CLI / TUI / MCP)                   │
│                                 │                                   │
│              Unix socket IPC    │                                    │
│                                 ▼                                   │
│                        zti-daemon                                  │
│                                                                     │
│  tree-sitter  ·  DSL chunking  ·  embedding  ·  ANN  ·  rerank    │
│  AST parsing     recursive       (Candle)    (usearch)  (Turbo)    │
│                                 │                                   │
│                                 ▼                                   │
│                        LanceDB Store                               │
│                  chunks · embeddings · file-snapshots              │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Extraction** — [tree-sitter](https://tree-sitter.github.io/) parses source code into ASTs. Language-specific queries extract nodes (functions, classes, structs, methods) and edges (calls, imports).

2. **Chunking** — Recursive DSL chunker splits code along semantic boundaries. Each chunk is a named scope with its context chain — not arbitrary line splits.

3. **Embedding** — On-device transformer models (via [Candle](https://github.com/huggingface/candle)) encode each chunk into a dense vector. Models download automatically from HuggingFace.

4. **Storage** — Everything lands in a local [LanceDB](https://lancedb.com/) database with vector search, file snapshots, and project metadata.

5. **Search** — ANN search via [usearch](https://github.com/unum-cloud/usearch) with optional exhaustive fallback, plus Turbo reranking for precision.

6. **Auto-Sync** — File watcher detects changes and incrementally re-indexes only the delta.

---

## Supported Platforms

Zebra is a **single Rust binary** — build from source on any target that Rust supports.

| Platform | Architectures | Status |
|----------|---------------|--------|
| **macOS** | x64, arm64 (Apple Silicon) | ✅ Full support |
| **Linux** | x64, arm64 | ✅ Full support |
| **Windows** | x64 | ✅ Full support |
| **FreeBSD** | x64 | ✅ Supported |
| **OpenBSD** | x64 | ✅ Supported |

---

## Supported Agents

Zebra exposes an MCP server — it works with **any MCP-compatible agent**. Tested and confirmed:

- **Claude Code** (Anthropic)
- **Cursor**
- **Codex CLI** (OpenAI)
- **opencode**
- **Hermes Agent**
- **Gemini CLI** (Google)
- **Antigravity IDE**
- **Kiro**
- **Pi**

Your agent gets seven tools: `searchQuery` (semantic symbol search), `searchPassage` (find similar code by example), `searchDep` (symbol lookup with call graph), `fileTree` (project structure), `projectList` (indexed projects), and `doctor` (diagnostics).

---

## Supported Languages & Formats

Each language gets a dedicated tree-sitter parser that extracts symbols and call edges:

| Language | Extensions | Symbol Extraction |
|----------|-----------|-------------------|
| **Rust** | `.rs` | functions, structs, enums, traits, impls, methods, macros, modules, type aliases |
| **TypeScript** | `.ts`, `.tsx` | functions, classes, methods, interfaces, type aliases, enums, JSX components |
| **JavaScript** | `.js`, `.jsx`, `.mjs`, `.cjs` | functions, classes, methods, arrow functions, JSX components |
| **Python** | `.py` | functions, classes, methods, async functions, decorators |
| **Dart** | `.dart` | functions, classes, methods, constructors, getters/setters |
| **Go** | `.go` | functions, methods, structs, interfaces, type aliases |
| **Solidity** | `.sol` | contracts, functions, events, modifiers, structs, enums |
| **OCaml** | `.ml`, `.mli` | functions (let/value), types, modules, classes, methods, module types |

---

## Supported Embedding Models

All models run **locally via Candle** — no API calls, no data leakage. Hardware auto-detection: Metal (macOS), CUDA (Linux/Windows), CPU fallback.

| Model | Params | Description |
|-------|--------|-------------|
| `all-MiniLM-L6-v2` | 22.7M | Lightweight baseline. High speed, minimal footprint. |
| `all-MiniLM-L12-v2` | 33.4M | Slightly deeper. Better accuracy than L6. |
| `bge-small-en-v1.5` | 33.4M | Gold standard for small models. Fast and accurate. |
| `bge-base-en-v1.5` | 109M | Excellent balance of speed and retrieval accuracy. |
| `bge-m3` | 567M | Heavyweight multi-lingual (100+ languages). GPU recommended. |
| `e5-small-v2` | 33.4M | Fast English-only. Requires `query:`/`passage:` prefixes. |
| `e5-base-v2` | 109M | Standard English-only E5. High accuracy. |
| `multilingual-e5-small` | 118M | Lightweight multilingual. Good for mixed-language data. |
| `multilingual-e5-base` | 278M | Heavy multilingual. Broad vocabulary, GPU recommended. |
| `gte-small` | 33.4M | Solid alternative to bge-small. Robust on diverse text. |
| `gte-base` | 109M | Competes with e5-base and bge-base. No prefixes required. |
| `gte-large` | 335M | Top-tier retrieval for English. VRAM heavy. |

---

## Hardware Acceleration

| Backend | Status | Notes |
|---------|--------|-------|
| **CUDA** | ✅ Supported | Linux & Windows. NVIDIA GPUs via Candle CUDA backend. |
| **Metal** | ✅ Supported | macOS. Apple Silicon & AMD GPUs via Candle Metal backend. |
| **CPU** | ✅ Supported | Always-available fallback. Works everywhere. |
| **Vulkan** | ❌ Not yet | Planned. Candle has experimental Vulkan support. |
| **RockX** | ❌ Not yet | AMD ROCm. Not currently integrated. |
| **NPU** | ❌ Not yet | Neural Processing Units (Apple Neural Engine, Qualcomm, Intel NPU). Not supported. |

Hardware is auto-probed at daemon startup and the fastest available backend is selected automatically.

---

## CLI Reference

```bash
zebraindex                              # Launch the terminal UI
zebraindex daemon --model <id>          # Start the background indexer daemon
zebraindex index -r <path>              # Index a project
zebraindex index -r <path> --refresh    # Force full re-index
zebraindex search -r <path> <query>     # Semantic search
zebraindex chat -r <path>               # Interactive search loop
zebraindex status [-r <path>]           # Show indexed project status
zebraindex doctor [-r <path>]           # Run diagnostics
zebraindex env                          # Show daemon environment info
zebraindex stop                         # Stop the daemon
zebraindex projects                     # List all indexed projects
zebraindex remove -r <path>             # Remove a project from the index
zebraindex --mcp                        # Run as MCP server (stdio)

# DSL debugging tools
zebraindex dsl -r <path> <subcommand>   # Dump DSL graph / dependency tree / project map
```

### Search options

```bash
zebraindex search -r <path> <query> \
  --limit 10 \            # Max results (default: 5)
  --lang rust,ts \        # Filter by language
  --glob "src/**/*.rs" \   # Filter by file pattern
  --mode passage \         # Search by example (default: query)
  --exhaustive             # Skip ANN, scan all chunks
```

---

## MCP Tools

When running as an MCP server, Zebra exposes these tools to agents:

| Tool | Purpose |
|------|---------|
| `searchQuery` | **Primary.** Search the codebase by intent in plain language. Returns full source code with file paths and line ranges — no follow-up file reads needed. |
| `searchPassage` | Find similar code by example. Paste a snippet or error message to locate related implementations. |
| `searchDep` | Look up a symbol by exact name. Returns kind, location, doc summary, callers/callees (to depth), and full source body. |
| `fileTree` | List project files and directory structure. Use instead of `find`, `ls -R`, or `glob`. |
| `projectList` | List all indexed projects with root paths. Use when unsure which project to target. |
| `doctor` | Run diagnostics on the embedding engine and index. Use when search tools return errors. |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Rust (edition 2024) |
| **Build** | Cargo workspace (24 crates) |
| **Parsing** | tree-sitter (9 language grammars) |
| **Embedding** | Candle (HuggingFace transformers on-device) |
| **Vector DB** | LanceDB (columnar, zero-copy reads) |
| **ANN Search** | usearch (HNSW graph) |
| **IPC** | Unix domain sockets (tokio async) |
| **MCP** | rmcp (stdio transport) |
| **TUI** | ratatui + crossterm |
| **CLI** | clap (derive API) |
| **Async Runtime** | tokio (multi-threaded) |
| **Parallelism** | rayon (CPU chunking & embedding) |

---

## Troubleshooting

**"Daemon not running"** — Start it with `zebraindex daemon --model <id>` or launch the TUI (`zebraindex`) which starts the daemon automatically.

**"No indexed projects found"** — Run `zebraindex index -r /path/to/project` first. Use `zebraindex projects` to see what's indexed.

**Search returns no results** — The project may use a language not yet supported. Check `zebraindex status` to see file counts. Try `--exhaustive` to bypass ANN.

**Indexing is slow on first run** — Expected. The initial backfill parses, chunks, and embeds every file. Subsequent runs are incremental and fast.

**Model download fails** — Check your internet connection. Models download from HuggingFace. Set `HF_HUB_CACHE` to control the cache location.

---

## License

MIT

---

<div align="center">

**Built with 🦀 Rust · Cargo · tree-sitter · Candle · LanceDB**

[Report Bug](https://github.com/hicaru/zebra_tree_indexer/issues) · [Request Feature](https://github.com/hicaru/zebra_tree_indexer/issues)

</div>
