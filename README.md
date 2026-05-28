<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://zebra.sh/hero-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://zebra.sh/hero-light.svg">
    <img src="https://zebra.sh/hero-light.svg" alt="Zebra — semantic code indexer for AI coding agents. On-device embedding, AST-aware chunking, incremental indexing, MCP server." width="100%" draggable="false"/>
  </picture>
</p>
<h1 align="center">Your AI agents deserve <em>semantic code search.</em></h1>

<p align="center">
  <strong>Star us&nbsp;❤️&nbsp;→</strong>&nbsp;<a href="https://github.com/hicaru/zebra_tree_indexer" title="Star Zebra on GitHub — open-source semantic code indexer for AI agents"><picture><source media="(prefers-color-scheme: dark)" srcset="https://zebra.sh/star-btn-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://zebra.sh/star-btn-light.svg"><img src="https://zebra.sh/star-btn-light.svg" alt="Star Zebra on GitHub — open-source semantic code indexer" height="36" align="absmiddle"/></picture></a> &nbsp;·&nbsp;
  <a href="https://zebra.sh" title="Visit zebra.sh — homepage"><picture><source media="(prefers-color-scheme: dark)" srcset="https://zebra.sh/zebra-inline-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://zebra.sh/zebra-inline-light.svg"><img src="https://zebra.sh/zebra-inline-light.svg" alt="zebra.sh — semantic code search for AI agents" height="36" align="absmiddle"/></picture></a> &nbsp;·&nbsp;
  <a href="https://t.me/+MLRSdmyS6CM4YzAy" title="Join the Zebra Telegram community"><picture><source media="(prefers-color-scheme: dark)" srcset="https://zebra.sh/telegram-inline-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://zebra.sh/telegram-inline-light.svg"><img src="https://zebra.sh/telegram-inline-light.svg" alt="Join the Zebra Telegram community" height="36" align="absmiddle"/></picture></a>
</p>

<p align="center"><b>Incremental</b> · only the delta &nbsp;·&nbsp; <b>On-device</b> · no external APIs &nbsp;·&nbsp; <b>Semantic</b> · search by meaning, not regex</p>

<div align="center">

[![github](https://img.shields.io/badge/github-hicaru/zebra__tree__indexer-181717?style=flat-square&logo=github)](https://github.com/hicaru/zebra_tree_indexer)
[![rust](https://img.shields.io/badge/rust-1.91+-db6d28?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-5B5BD6?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![telegram](https://img.shields.io/badge/telegram-joint-26A5E4?style=flat-square&logo=telegram)](https://t.me/+MLRSdmyS6CM4YzAy)

</div>

<br/>

<h2 align="center">What is Zebra?</h2>

<p align="center">
  Zebra indexes your codebase locally — parsing source files with tree-sitter, embedding each symbol and chunk with an on-device transformer model (via Candle), and storing everything in LanceDB — so you can <strong>search code by meaning, not by string matching.</strong>  
</p>

<p align="center">
  It runs entirely on your machine. No data leaves your computer. No API keys. No cloud dependency.
</p>

<br/>

<h2 align="center">Search by meaning</h2>

<p align="center">
  <code>searchQuery "user session validation"</code> — finds the actual session validation code,<br/>
  not just files containing the literal string "user session validation".<br/>
  <b>Semantic embedding</b> + <b>keyword boost</b> + <b>Turbo reranking</b> = precise results.
</p>

<p align="center">
  Two modes: <b>Query</b> searches symbols and code by meaning. <b>Passage</b> finds similar code passages from a snippet or error message.
</p>

<br/>

<h2 align="center">Get started</h2>

Install via Cargo:

```sh
cargo install --path crates/apps/zebraindex/
```

Or build from source:

```sh
git clone https://github.com/hicaru/zebra_tree_indexer.git
cd zebra_tree_indexer
cargo build --release -p zebraindex
```

### Quick start

Start the TUI — it walks you through model selection, downloads, daemon launch, and indexing:

```sh
zebraindex
```

Index a project from the CLI:

```sh
zebraindex index -r /path/to/your/project
```

Search across an indexed project:

```sh
zebraindex search -r /path/to/your/project "rate limiting middleware"
```

Chat interactively:

```sh
zebraindex chat -r /path/to/your/project
```

### MCP server for AI coding agents

Run as an MCP server so Claude Code, Cursor, and other MCP-aware agents can search your codebase:

```sh
zebraindex --mcp
```

Add to your **Claude Code / Cursor** config:

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

Or register it directly with Claude via CLI:

```sh
claude mcp add -s user zebra-mcp -- zebraindex --mcp
```

Add to **opencode** (`~/.config/opencode/opencode.json`):

```json
"zebra-mcp": {
  "type": "local",
  "command": ["zebraindex", "--mcp"],
  "enabled": true
}
```

Your agent gets four tools: `searchQuery` (semantic symbol search), `searchPassage` (find similar code by example), `fileTree` (project structure), `projectList` (indexed projects), and `doctor` (diagnostics).

<br/>

<h2 align="center">Commands</h2>

| Command | Description |
|---------|-------------|
| `zebraindex` | Launch the terminal UI (model selection, daemon, search) |
| `zebraindex daemon --model <id>` | Start the background indexer daemon |
| `zebraindex index -r <path>` | Index a project |
| `zebraindex search -r <path> <query>` | Semantic search |
| `zebraindex chat -r <path>` | Interactive search loop |
| `zebraindex status` | Show indexed project status |
| `zebraindex doctor` | Run diagnostics |
| `zebraindex env` | Show daemon environment info |
| `zebraindex stop` | Stop the daemon |
| `zebraindex projects` | List all indexed projects |
| `zebraindex remove -r <path>` | Remove a project from the index |
| `zebraindex --mcp` | Run as MCP server (stdio) |

<br/>

<h2 align="center">Incremental engine</h2>

<p align="center">
  Zebra tracks file snapshots and only re-indexes what changed.<br/>
  Run once to backfill. Re-run anytime — unchanged files skip straight to cache.<br/>
  <b>Δ-only indexing.</b>
</p>

<br/>

<h2 align="center">Supported languages</h2>

<p align="center">
  <b>Rust</b>&nbsp;·&nbsp;<b>TypeScript</b>&nbsp;·&nbsp;<b>JavaScript</b>&nbsp;·&nbsp;<b>Python</b>&nbsp;·&nbsp;<b>Dart</b>&nbsp;·&nbsp;<b>Go</b>&nbsp;·&nbsp;<b>Solidity</b>
</p>

<p align="center">
  Each language gets a tree-sitter parser that extracts symbols (functions, classes, structs, enums, methods, fields, etc.) and call edges — enabling semantic understanding beyond plain text.
</p>

<br/>

<h2 align="center">Supported embedding models</h2>

<p align="center">
  All models run locally via Candle (no API calls, no data leakage).<br/>
  Hardware auto-detection: Metal (macOS), CUDA (Linux/Windows), CPU fallback.
</p>

<table align="center" width="100%">
  <tr>
    <th>Model</th>
    <th>Parameters</th>
    <th>Description</th>
  </tr>
  <tr><td><code>all-MiniLM-L6-v2</code></td><td>22.7M</td><td>Lightweight baseline. High speed, minimal footprint.</td></tr>
  <tr><td><code>all-MiniLM-L12-v2</code></td><td>33.4M</td><td>Slightly deeper. Better accuracy than L6.</td></tr>
  <tr><td><code>bge-small-en-v1.5</code></td><td>33.4M</td><td>Gold standard for small models. Fast and accurate.</td></tr>
  <tr><td><code>bge-base-en-v1.5</code></td><td>109M</td><td>Excellent balance of speed and retrieval accuracy.</td></tr>
  <tr><td><code>bge-m3</code></td><td>567M</td><td>Heavyweight multi-lingual (100+ languages). GPU recommended.</td></tr>
  <tr><td><code>e5-small-v2</code></td><td>33.4M</td><td>Fast English-only. Requires <code>query:</code>/<code>passage:</code> prefixes.</td></tr>
  <tr><td><code>multilingual-e5-small</code></td><td>118M</td><td>Lightweight multilingual. Good for mixed-language data.</td></tr>
  <tr><td><code>gte-small</code></td><td>33.4M</td><td>Solid alternative to bge-small. Robust on diverse text.</td></tr>
  <tr><td><code>gte-base</code></td><td>109M</td><td>Competes with e5-base and bge-base. No prefixes required.</td></tr>
</table>

<br/>

<h2 align="center">Architecture</h2>

<p align="center"><b>Daemon</b> (Rust, background process with Unix socket IPC)</p>
<pre align="center">
  ┌─────────┐     ┌─────────────┐     ┌──────────┐     ┌──────────┐
  │  CLI /  │────▶│   Daemon    │────▶│  Embed   │────▶│ LanceDB  │
  │  TUI /  │◀────│ (Unix sock) │◀────│  Engine  │◀────│  Store   │
  │  MCP    │     │             │     │ (Candle) │     │          │
  └─────────┘     │  tree-sitter│     └──────────┘     └──────────┘
                  │  · Pipeline │     ┌──────────┐
                  │  · ANN      │────▶│  Rerank  │
                  │  · Rerank   │     │ (Turbo)  │
                  └─────────────┘     └──────────┘
</pre>

<p align="center">
  <b>Zebra</b> &nbsp;·&nbsp; <b>tree-sitter</b> AST parsing → DSL chunking → on-device embedding → ANN search → Turbo reranking
</p>

<br/>

<h2 align="center">Why Zebra?</h2>

<table width="100%">
  <tr>
    <td align="center" width="33%"><b>🔒 Private</b><br/>Runs on-device. No data ever leaves your machine. No API calls.</td>
    <td align="center" width="33%"><b>⚡ Incremental</b><br/>Only re-indexes changed files. Δ-only. No full rebuilds.</td>
    <td align="center" width="33%"><b>🧠 Semantic</b><br/>Understands code by meaning. Finds what you meant, not just what you typed.</td>
  </tr>
  <tr>
    <td align="center"><b>🔌 MCP-native</b><br/>First-class MCP server. Works with Claude Code, Cursor, and any MCP client.</td>
    <td align="center"><b>📦 Self-contained</b><br/>Single binary. Downloads models from HuggingFace automatically.</td>
    <td align="center"><b>🦀 Rust core</b><br/>Zero-cost abstractions. Parallel chunking. Production-grade.</td>
  </tr>
</table>

<br/>

<p align="center">
  Zebra is purpose-built for AI coding agents — an MCP-native semantic code indexer<br/>
</p>

<br/><br/>

<p align="center">
  <b>Built with 🦀 Rust</b>
</p>

<p align="center"><sub>MIT OR Apache-2.0 · © Zebra contributors</sub></p>

