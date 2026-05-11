# research-orchestrator

A stateful MCP server that other AI agents use as a **research brain**.

Most MCP servers are thin wrappers — a search tool, a file reader, fifty lines of glue code. This one is different: it runs a full research pipeline internally, exposes six clean tools to any MCP client, and handles scoping, fetching, conflict detection, and synthesis so the calling agent doesn't have to.

```
calling agent
    │
    │  start_research(topic, depth)
    │  get_state(job_id)
    │  get_synthesis(job_id)
    ▼
research-orchestrator  ←── stateful job + state machine
    │
    ├── ScopeAgent      parse question → source targets
    ├── SourceAgent ×N  fetch + summarize in parallel
    ├── ConflictAgent   detect contradictions between sources
    ├── GapAgent        identify what's still unknown
    └── SynthesisAgent  assemble answer + confidence scores
```

---

## How it works

Every research job moves through a guarded state machine:

```
SCOPING → GATHERING → CONFLICTING → SYNTHESIZING → DONE
                                                  ↘ FAILED (any stage)
```

Each stage dispatches one or more scoped Claude API calls (subagents). No subagent sees more context than it needs. A context manager chunks and trims findings as they accumulate so no single call approaches the token limit — the fix for context degradation on long-running agentic tasks.

`ask_followup()` forks child jobs that inherit the parent's findings, so drilling deeper on a sub-question costs a fraction of starting fresh.

---

## Quickstart

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), an [Anthropic API key](https://console.anthropic.com).

```bash
git clone https://github.com/wokdabat/research-orchestrator
cd research-orchestrator

uv sync

cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
```

**Run in stdio mode** (Claude Desktop, Claude Code):
```bash
uv run server.py
```

**Run in HTTP mode** (n8n, custom callers — port 8000):
```bash
uv run server.py --http
```

**Run tests** (no API key required):
```bash
uv run pytest tests/ -v
```

---

## Connecting to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "research-orchestrator": {
      "command": "uv",
      "args": ["run", "server.py"],
      "cwd": "/path/to/research-orchestrator",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Restart Claude Desktop. The six tools will appear automatically.

---

## The six MCP tools

| Tool | Description |
|------|-------------|
| `start_research(topic, depth)` | Start a job. Returns `job_id`. Pipeline runs async. |
| `get_state(job_id)` | Poll current state, findings count, token usage. |
| `inject_source(job_id, content, source_url)` | Feed in your own content during GATHERING. |
| `get_conflicts(job_id)` | Return detected contradictions with confidence scores. |
| `get_synthesis(job_id)` | Return final answer, per-claim confidence, open questions. |
| `ask_followup(job_id, question, depth)` | Fork a child job inheriting parent findings. |

`depth` ranges from 1 (quick, 2–3 sources) to 5 (exhaustive, up to 10 sources).

---

## Example — calling from Python

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/mcp") as client:

        # Start a research job
        result = await client.call_tool("start_research", {
            "topic": "How does Kling v2 handle temporal consistency in video generation?",
            "depth": 3
        })
        job_id = result[0].text  # parse JSON as needed

        # Poll until done
        import json, time
        while True:
            state = await client.call_tool("get_state", {"job_id": job_id})
            data = json.loads(state[0].text)
            print(f"State: {data['state']}  |  Findings: {data['findings_count']}")
            if data["state"] in ("DONE", "FAILED"):
                break
            time.sleep(3)

        # Get the answer
        synthesis = await client.call_tool("get_synthesis", {"job_id": job_id})
        print(json.loads(synthesis[0].text)["synthesis"])

asyncio.run(main())
```

---

## Example — calling from n8n

Set up an HTTP Request node:

```
Method:  POST
URL:     http://localhost:8000/mcp
Body:    {
           "jsonrpc": "2.0",
           "method": "tools/call",
           "params": {
             "name": "start_research",
             "arguments": {
               "topic": "{{ $json.research_topic }}",
               "depth": 3
             }
           },
           "id": 1
         }
```

Follow with a loop node polling `get_state` every 5 seconds until `DONE`, then call `get_synthesis`.

---

## Project structure

```
research-orchestrator/
├── server.py        MCP server — 6 tools, async pipeline, job store
├── models.py        Job dataclass + guarded state machine
├── agents.py        5 scoped Claude API subagents
├── fetchers.py      DuckDuckGo search + BeautifulSoup content extraction
├── tests/
│   └── test_unit.py 22 unit tests (no network, no API key needed)
├── pyproject.toml
├── .env.example
└── .vscode/         VS Code interpreter + launch configs
```

---

## Configuration

All config via `.env` (copy from `.env.example`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Your Anthropic API key |
| `RESEARCH_MCP_MODEL` | No | `claude-sonnet-4-20250514` | Model used by subagents |
| `RESEARCH_MCP_PORT` | No | `8000` | HTTP transport port |
| `RESEARCH_MCP_MAX_CONCURRENT` | No | `5` | Parallel fetch limit |

---

## Swapping the web fetcher

`fetchers.py` exposes two functions with a stable interface:

```python
async def search_and_fetch(query: str, max_results: int = 5) -> list[FetchResult]: ...
async def fetch_url(url: str) -> FetchResult: ...
```

To swap in Tavily or Brave Search, implement the same signatures returning `FetchResult` objects — nothing else in the codebase needs to change.

---

## Architecture notes

**Why a state machine?**
Explicit states with guarded transitions mean the pipeline can never silently skip a stage, double-run synthesis, or return partial results. Every failure lands in `FAILED` with an error message. The state is persisted in memory (swappable to Redis or SQLite) so jobs survive across tool calls.

**Why scoped subagents?**
Each Claude API call gets exactly the context it needs — a SourceAgent sees one page, a ConflictAgent sees a compact claim list, SynthesisAgent sees summaries not raw HTML. This keeps individual call costs low and prevents the context degradation that plagues long-running agentic tasks.

**Why DuckDuckGo HTML scraping?**
Zero dependencies, no API key, works out of the box. The tradeoff is occasional bot detection (403s) on individual results — the fetcher handles this gracefully by skipping failed results and moving to the next. For production use, swap to Tavily via the interface above.

---

## Stack

- [FastMCP](https://github.com/jlowin/fastmcp) — MCP server framework
- [Anthropic Python SDK](https://github.com/anthropic/anthropic-sdk-python) — subagent Claude calls
- [httpx](https://www.python-httpx.org/) — async HTTP fetching
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) + lxml — content extraction
- [python-dotenv](https://github.com/theskumar/python-dotenv) — env var loading

---

## License

MIT
