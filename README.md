# Chronicle — Personal AI Analyst

Chronicle connects to your real data — Spotify, GitHub, finances, fitness records, journal entries — and tells you what it says about you that you haven't admitted yet. Five specialised AI agents run concurrently, each with a locked inference tier, deployment configuration, and OOM safety check.

This is a multi-session build. Each session extends the previous one without removing anything.

---

## Quick Start

**You need one thing before anything else: a Gemini API key.**

Get one free at [aistudio.google.com](https://aistudio.google.com) → "Get API key" → Create. It's free with generous limits.

Then copy `.env.example` to `.env` and paste in your key:

```
cp .env.example .env
```

```
GEMINI_API_KEY=your_actual_key_here
```

`.env` is git-ignored on purpose — never commit it. That's the only external step. Everything else is handled below.

---

## Option A — Local Setup (Python)

**Requirements:** Python 3.11 or later. Check with `python3 --version`.

### Step 1 — Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 2 — Activate it

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Your prompt will now show `(.venv)`.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: FastAPI, uvicorn, aiohttp, pydantic, python-dotenv, certifi. (Inference calls the Gemini REST API directly — no Google SDK dependency needed.)

### Step 4 — Add your API key

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_actual_key_here
```

### Step 5 — Run the verification

```bash
python agent.py
```

Expected output:

```
╔══════════════════════════════════════════════════════╗
║  Chronicle — Session 11.3 Startup                    ║
╚══════════════════════════════════════════════════════╝

  ── OOM Prevention Check ──────────────────────────────

  ✓ ingestion    GPU: L4       max_model_len:  4,096  max_concurrent:   1  KV/req: 0.500 GB
  ✓ pattern      GPU: L4       max_model_len:  4,096  max_concurrent:   1  KV/req: 0.500 GB
  ✓ timeline     GPU: L4       max_model_len:  4,096  max_concurrent:   1  KV/req: 0.500 GB
  ✓ brutality    GPU: A100-40  max_model_len:  8,192  max_concurrent:   5  KV/req: 1.857 GB
  ✓ synthesis    GPU: A100-40  max_model_len:  8,192  max_concurrent:   5  KV/req: 1.857 GB

  OOM PREVENTION: PASS — all agents have safe concurrent capacity

  ── Verification: 5/5 checks passed ────────────────────

  ✓ Session 11.3 COMPLETE. Chronicle is deployment-ready.
    Start the API: python api.py
```

If all 5 checks pass, proceed.

### Step 6 — Start the server

```bash
python api.py
```

### Step 7 — Open the UI

Go to: **http://localhost:8000**

The dashboard, agent cards, and chat interface will load. Type a question and click Analyse.

---

## Option B — Docker Setup

**Requirements:** Docker Desktop installed and running. Check with `docker --version`.

### Step 1 — Add your API key

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_actual_key_here
```

### Step 2 — Build and start

```bash
docker compose up --build
```

Docker will pull the Python base image, install all dependencies, and start the server. First build takes ~60 seconds. Subsequent starts take ~3 seconds.

### Step 3 — Open the UI

Go to: **http://localhost:8000**

To stop:

```bash
docker compose down
```

To rebuild after code changes:

```bash
docker compose up --build
```

---

## Verifying Everything Works

Once the server is running, you can check each endpoint directly:

| URL | What it returns |
|-----|-----------------|
| `http://localhost:8000` | The Chronicle UI |
| `http://localhost:8000/health` | Session version, OOM status, all agent configs |
| `http://localhost:8000/docs` | Swagger UI — interactive docs for all 10 endpoints |
| `http://localhost:8000/vram-budget/tiered` | Per-agent VRAM breakdown across S11.1/11.2/11.3 |
| `http://localhost:8000/oom-check` | OOM prevention pass/fail per agent |
| `http://localhost:8000/deployment-config` | Full `vllm serve` launch command per agent |
| `http://localhost:8000/cost-model` | 4 GPU cost scenarios with annual savings |
| `http://localhost:8000/concurrency-table` | How context window size affects concurrent capacity |
| `http://localhost:8000/survivability` | Which tasks survive INT4 quantization |

---

## What Was Built — Session by Session

### Session 11.1 — Inference Foundation

**Goal:** Get all 5 Chronicle agents firing concurrently against a real AI API and measure the performance baseline.

**What was built:**

- `CHRONICLE_AGENTS` — the 5 permanent agents defined with their roles and tiers:
  - `ingestion` — parses and normalises raw data from all sources
  - `pattern` — finds cross-source correlations
  - `timeline` — sequences life events chronologically
  - `brutality` — delivers honest analysis without softening
  - `synthesis` — produces the final structured analyst brief

- `calculate_chronicle_vram_budget()` — calculates total VRAM needed for all 5 agents at a given precision (FP16, INT4, etc.). Establishes the S11.1 baseline: **90 GB** at uniform FP16.

- `chronicle_infer()` — fires a single async inference request against the Gemini REST API and measures Time to First Token (TTFT) and Time Per Output Token (TPOT).

- `run_concurrent_analysis()` — dispatches all 5 agents simultaneously using `asyncio` + `aiohttp`. All agents fire at the same moment. Wall clock time reflects true concurrent load.

- `BenchmarkResult` / `AnalysisRequest` — Pydantic schemas that remain permanent through all sessions.

- **API endpoints added:** `GET /health`, `POST /analyze`, `GET /vram-budget`

- **Dashboard:** Split layout with agent status card, inference metrics card, and VRAM budget card with precision selector.

**Key result:** 5 agents fire concurrently in a single wall-clock window. TTFT measured across all agents.

---

### Session 11.2 — Model Quantization

**Goal:** Assign the right precision to each agent based on whether its task survives quantization. Not every agent needs full FP16.

**What was built:**

- `CHRONICLE_AGENTS` extended with per-agent fields:
  - `precision` — `int4` for utility agents, `fp16` for frontier agents
  - `model_size_b` — 7B for utility, 13B for frontier
  - `gpu_tier` — `L4` for utility, `A100-40` for frontier
  - `monthly_gpu_cost_usd` — $450 (L4), $1,500 (A100-40)
  - `survivability_note` — why this precision is safe for this task

- `TASK_SURVIVABILITY_MATRIX` — 11 task types tested at INT4. Results:
  - **Survives INT4 (≥90% retention):** intent classification, NER, sentiment, summarisation, data parsing, temporal sequencing, cross-source correlation
  - **Requires FP16 (<90% retention):** structured generation, long-context coherence, multi-constraint reasoning, code generation

- `calculate_tiered_vram_budget()` — replaces the uniform budget with per-agent precision. Reduced from 90 GB to ~84 GB.

- `calculate_monthly_gpu_cost()` — 3 GPU deployment scenarios:
  - **Scenario A:** All A100-80, no tiering → $9,375/mo
  - **Scenario B:** 3× L4 (utility) + 2× A100-40 (frontier) → $4,350/mo, saves $60,300/yr
  - **Scenario C:** 3× A10G + 2× A100-40 → $4,650/mo

- `task_survivability_matrix()` — queryable by task type.

- `chronicle_infer()` updated with tier-aware prompts: utility agents get structured 2-sentence prompts, frontier agents get full analytical prompts.

- **API endpoints added:** `GET /vram-budget/tiered`, `GET /cost-model`, `GET /survivability`, `GET /calibration-stats`

- **Dashboard:** Precision badges on agent cards (INT4 green, FP16 purple), tiered VRAM card, cost model card with 3 scenarios.

**Key result:** VRAM dropped from 90 GB to ~84 GB. Monthly GPU cost halved vs naive all-A100 setup.

---

### Session 11.3 — GPU Resource Allocation (Current)

**Goal:** Lock the exact deployment configuration that prevents Chronicle from crashing at 2 AM. Every number calculated here goes into the actual `vllm serve` command.

**What was built:**

- `CHRONICLE_AGENTS` extended with:
  - `max_model_len` — 4,096 for utility agents, 8,192 for frontier agents. Without this lock, Llama-3 defaults to 128K context, consuming 64 GB KV cache per agent.
  - `gpu_memory_utilization` — 0.28 for utility (co-located on shared L4), 0.85 for frontier (dedicated A100-40 with 15% safety buffer)

- `GPU_VRAM_GB` — reference dict for all 6 GPU tiers (T4→H100-80).

- `calculate_max_safe_concurrent()` — the OOM prevention formula:
  ```
  Max Safe Concurrent = (Effective VRAM - Weights - Overhead - Buffer) / KV_per_request
  ```
  Results: utility agents handle 1 concurrent request each on their L4 partition. Frontier agents handle 5 concurrent requests each on their A100-40.

- `oom_prevention_check()` — runs the formula for all 5 agents at startup. If any agent returns 0 concurrent slots, Chronicle refuses to start. The crash is caught at deploy time, not at 2 AM.

- `vllm_config_per_agent()` — generates the exact `vllm serve` command for each agent, including `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, `--tensor-parallel-size`, and port assignments (8100–8104).

- `colocation_partitioner()` — validates the 3 utility agents fit on one shared L4:
  - 3 × 0.28 = 0.84 model fraction + 0.08 system overhead = **0.92 total** (safe, ≤ 1.0)
  - Remaining 1.9 GB headroom

- `kv_cache_growth_simulator()` — simulates KV cache VRAM growth under a given requests-per-minute rate. Shows the exact minute OOM would occur without the concurrent request guard.

- `calculate_tiered_vram_budget()` updated — KV cache now calibrated to per-agent `max_model_len`. Utility agents locked at 4K (2.0 GB KV each) instead of the conservative 8K estimate from S11.2, saving 6 GB total.

- `calculate_monthly_gpu_cost()` updated — **Scenario D added (co-location):**
  - 1× L4 shared by 3 utility agents + 2× A100-40 for frontier → **$3,450/mo**
  - Saves $10,800/yr vs S11.2's separate-GPU approach
  - Saves $71,100/yr vs naive all-A100 setup

- `chronicle_infer()` updated — input length guard added. Requests longer than the agent's `max_model_len` are rejected before dispatch with a clear error message.

- **API endpoints added:** `GET /deployment-config`, `GET /oom-check`, `GET /concurrency-table`

- **Dashboard:** Deployment config card (per-agent mml / util / concurrent slots), OOM safety card (✓ ALL AGENTS SAFE), `mml:` badge on agent cards.

**Session 11.3 verification — 5/5 checks:**
1. All 5 agents have `max_model_len` and `gpu_memory_utilization` set
2. OOM prevention passes: all agents have `max_safe_concurrent > 0`
3. S11.3 calibrated VRAM (78.2 GB) < S11.2 conservative estimate (84.2 GB) — saves 6 GB
4. Co-location partition valid: grand total 0.92 ≤ 1.0
5. Scenario D ($3,450/mo) < Scenario B ($4,350/mo) — co-location wins

**VRAM journey across Week 11:**
```
S11.1 uniform FP16 (no tiering):       90.0 GB
S11.2 tiered precision (8K budget):    84.2 GB   saved  5.8 GB
S11.3 calibrated max_model_len:        78.2 GB   saved 11.8 GB total
```

---

## What's Coming — Upcoming Sessions

### Session 12.1 — FastAPI Gateway + MCP Ingestion

Chronicle stops using placeholder data and connects to your real accounts.

- `MCP_SERVERS` dict mapping each data source to its MCP server endpoint
- `MCPIngestionClient` — async client that pulls live structured records from each source using the Model Context Protocol
- Sources: **Spotify** (listening history), **GitHub** (commit patterns), **Finance** (transaction records), **Fitness** (activity logs), **Journal** (personal entries)
- The Ingestion agent will pull live data before generating its analysis prompt
- `/analyze` updated to trigger MCP pulls before agent dispatch
- Dashboard: MCP Connectors card showing live connection status per source

### Session 12.2 — SSE Streaming

Chronicle stops waiting for all 5 agents to finish before showing anything.

- `/analyze` replaced with a Server-Sent Events streaming endpoint
- Tokens stream from each agent as they arrive — no more waiting for the slowest agent
- Per-agent streaming indicators in the dashboard
- Real TTFT measurement (first token, not first response)

### Session 12.3 — Async Job Queue

Deep analyses that take longer than 30 seconds get queued properly.

- `POST /analyze` returns `202 Accepted` with a job ID immediately
- `GET /jobs/{id}` polls for result
- Background worker processes the queue
- No more HTTP timeouts on long analyses

### Session 13.1 — OpenTelemetry Tracing

Every agent request becomes a traceable span.

- OTel instrumentation on all 5 agents
- Distributed trace per analysis: one root span, 5 child spans (one per agent)
- Trace viewer card in the dashboard showing per-agent latency breakdown
- Export to any OTel-compatible backend (Jaeger, Grafana Tempo, etc.)

### Session 14.1 — Semantic Caching

Reduce inference cost by catching semantically similar questions.

- Embedding-based cache: if a new question is >90% similar to a cached one, return the cached result
- Cache hit rate tracked per agent
- Reduces effective GPU-hours by 30–60% in practice

### Session 14.2 — Per-Agent Spend Ledger

Know exactly what each agent costs per question, per day, per month.

- Token counting per agent per request
- Cost attribution: $X per question broken down by agent
- Monthly spend projection card in the dashboard
- Alert threshold: flag when spend exceeds a per-agent daily budget

---

## Project Structure

```
session3/
├── agent.py          # Inference core: agents, VRAM, OOM, vLLM config, cost model
├── api.py            # FastAPI server: all HTTP endpoints
├── index.html        # Dashboard UI: chat + live metrics cards
├── requirements.txt  # Python dependencies
├── .env              # API key (never commit this)
├── Dockerfile        # Container build
└── docker-compose.yml # Multi-service orchestration
```

`agent.py` is the source of truth. Every number in `api.py` and `index.html` comes from functions defined there. Sessions extend these files — nothing is ever removed.

---

## Endpoints Reference

| Method | Path | Session | Description |
|--------|------|---------|-------------|
| `GET` | `/` | 11.1 | Chronicle UI |
| `GET` | `/health` | 11.1 | Version, OOM status, agent configs |
| `POST` | `/analyze` | 11.1 | Run all 5 agents concurrently |
| `GET` | `/vram-budget` | 11.1 | Uniform VRAM at a given precision |
| `GET` | `/vram-budget/tiered` | 11.2 | Per-agent tiered VRAM breakdown |
| `GET` | `/cost-model` | 11.2 | 4 GPU deployment cost scenarios |
| `GET` | `/survivability` | 11.2 | INT4 task survivability matrix |
| `GET` | `/deployment-config` | 11.3 | vLLM launch commands per agent |
| `GET` | `/oom-check` | 11.3 | OOM prevention check per agent |
| `GET` | `/concurrency-table` | 11.3 | Context window vs concurrent capacity |

---

## Troubleshooting

**`GEMINI_API_KEY environment variable is not set`**
Open `.env` and make sure the key is set with no quotes and no spaces around `=`:
```
GEMINI_API_KEY=AIza...your_key_here
```

**`address already in use` on port 8000**
Something is already running on port 8000. Kill it:
```bash
# macOS / Linux
lsof -ti :8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```
Then restart with `python api.py`.

**SSL certificate error on macOS**
This is handled automatically via `certifi`. If it still appears, run:
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```
Replace `3.x` with your Python version.

**Dashboard cards show "API offline"**
The UI is running but can't reach the API. Make sure `python api.py` (or `docker compose up`) is running, then refresh the page.

**Docker: `Cannot connect to the Docker daemon`**
Docker Desktop is not running. Open Docker Desktop from your Applications folder and wait for it to start (the whale icon in the menu bar stops animating when ready), then re-run `docker compose up --build`.
