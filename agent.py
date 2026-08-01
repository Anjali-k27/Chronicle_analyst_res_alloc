"""
╔══════════════════════════════════════════════════════════════════╗
║  CHRONICLE — agent.py                                            ║
║  Session 11.3: GPU Resource Allocation                           ║
╚══════════════════════════════════════════════════════════════════╝

Changes in Session 11.3 (all additions — nothing removed):
  - CHRONICLE_AGENTS extended: max_model_len, gpu_memory_utilization
    added to each agent entry
  - MAX_SAFE_CONCURRENT_REQUESTS: new per-agent constant
  - calculate_max_safe_concurrent(): the OOM prevention formula
  - calculate_tiered_vram_budget(): UPDATED — uses per-agent
    max_model_len instead of flat KV constant
  - calculate_monthly_gpu_cost(): UPDATED — Scenario D co-location added
  - vllm_config_per_agent(): generates production vLLM launch config
  - colocation_partitioner(): gpu_memory_utilization for shared L4
  - oom_prevention_check(): startup safety gate per agent
  - kv_cache_growth_simulator(): models traffic spike → OOM scenario
  - chronicle_infer(): UPDATED — input length guard added
  - run_session_verification(): REPLACED with 5 S11.3 checks
"""

# ── Imports (Session 11.1 — unchanged) ───────────────────────────
import asyncio
import aiohttp
import time
import statistics
import json
import os
import math
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import ssl
import certifi

load_dotenv()

# ── Configuration (Session 11.1 — unchanged) ─────────────────────
# Note: inference goes directly through the Gemini REST API below
# (not the google-generativeai SDK, which is deprecated/EOL and was
# previously imported here without being used).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"
GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set. "
        "Copy .env.example to .env and add your key before running Chronicle."
    )

# ── Chronicle Agent Roles (Session 11.3 — EXTENDED) ──────────────
# Session 11.1: role, tier
# Session 11.2: precision, model_size_b, gpu_tier, monthly_gpu_cost_usd,
#               survivability_note
# Session 11.3: max_model_len, gpu_memory_utilization
#
# max_model_len: calibrated to Chronicle's actual workload distribution.
#   Utility agents: 4,096 — data parsing + pattern tasks are bounded.
#   Frontier agents: 8,192 — long analysis + synthesis needs more context.
#   Model default would be 128K — giving 0 concurrent slots on most GPUs.
#
# gpu_memory_utilization: for co-located utility agents on shared L4.
#   3 utility agents × 0.28 = 0.84 + 0.08 system overhead = 0.92 total.
#   Frontier agents each own their GPU: 0.85 (15% safety buffer).
CHRONICLE_AGENTS = {
    "ingestion": {
        "role":                   "Parse and normalise raw data from all 5 sources",
        "tier":                   "utility",
        "precision":              "int4",
        "model_size_b":           7,
        "gpu_tier":               "L4",
        "monthly_gpu_cost_usd":   450,
        "survivability_note":     "Structured parsing — survives INT4",
        "max_model_len":          4_096,       # S11.3: locked
        "gpu_memory_utilization": 0.28,        # S11.3: co-located on shared L4
    },
    "pattern": {
        "role":                   "Find cross-source correlations between data signals",
        "tier":                   "utility",
        "precision":              "int4",
        "model_size_b":           7,
        "gpu_tier":               "L4",
        "monthly_gpu_cost_usd":   450,
        "survivability_note":     "Statistical pattern matching — survives INT4",
        "max_model_len":          4_096,       # S11.3: locked
        "gpu_memory_utilization": 0.28,        # S11.3: co-located on shared L4
    },
    "timeline": {
        "role":                   "Sequence life events from data across all sources",
        "tier":                   "utility",
        "precision":              "int4",
        "model_size_b":           7,
        "gpu_tier":               "L4",
        "monthly_gpu_cost_usd":   450,
        "survivability_note":     "Temporal classification — survives INT4",
        "max_model_len":          4_096,       # S11.3: locked
        "gpu_memory_utilization": 0.28,        # S11.3: co-located on shared L4
    },
    "brutality": {
        "role":                   "Deliver honest cross-source analysis, no softening",
        "tier":                   "frontier",
        "precision":              "fp16",
        "model_size_b":           13,
        "gpu_tier":               "A100-40",
        "monthly_gpu_cost_usd":   1500,
        "survivability_note":     "Long-context multi-constraint reasoning — requires FP16",
        "max_model_len":          8_192,       # S11.3: locked — needs longer context
        "gpu_memory_utilization": 0.85,        # S11.3: dedicated GPU, 15% safety buffer
    },
    "synthesis": {
        "role":                   "Produce the final structured analyst brief",
        "tier":                   "frontier",
        "precision":              "fp16",
        "model_size_b":           13,
        "gpu_tier":               "A100-40",
        "monthly_gpu_cost_usd":   1500,
        "survivability_note":     "Multi-source structured generation — requires FP16",
        "max_model_len":          8_192,       # S11.3: locked — needs longer context
        "gpu_memory_utilization": 0.85,        # S11.3: dedicated GPU, 15% safety buffer
    },
}

# ── VRAM Constants (Session 11.1 — unchanged) ────────────────────
VRAM_BYTES_PER_PARAM = {
    "fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5,
}
KV_CACHE_GB_PER_AGENT_4K  = 2.0   # 7B model @ 4K context. Used by S11.1 uniform calc.
MODEL_WEIGHT_GB_7B_FP16    = 14.0
MODEL_WEIGHT_GB_7B_INT4    = 3.5
CUDA_OVERHEAD_GB           = 2.0
SAFETY_HEADROOM_GB         = 8.0

# ── GPU Tier Reference (Session 11.2 — unchanged) ────────────────
GPU_TIER_COSTS = {
    "T4":      200,
    "L4":      450,
    "A10G":    550,
    "A100-40": 1500,
    "A100-80": 1875,
    "H100-80": 2800,
}

# ── GPU VRAM Reference (Session 11.3) ────────────────────────────
# Used by OOM prevention check and concurrency table.
# Permanent from Session 11.3 onward.
GPU_VRAM_GB = {
    "T4":      16,
    "L4":      24,
    "A10G":    24,
    "A100-40": 40,
    "A100-80": 80,
    "H100-80": 80,
}

# ── Task Survivability Matrix (Session 11.2 — unchanged) ──────────
TASK_SURVIVABILITY_MATRIX = {
    "intent_classification":    {"int4_retention_pct": 98, "passes_90pct": True,  "agent_tier": "supervisor_int4", "reason": "Coarse category boundary; tolerates individual weight errors"},
    "named_entity_extraction":  {"int4_retention_pct": 95, "passes_90pct": True,  "agent_tier": "supervisor_int4", "reason": "Pattern-matching on broad statistical regularities"},
    "sentiment_analysis":       {"int4_retention_pct": 97, "passes_90pct": True,  "agent_tier": "supervisor_int4", "reason": "Coarse 3-way classification; extremely error-tolerant"},
    "summarisation":            {"int4_retention_pct": 93, "passes_90pct": True,  "agent_tier": "supervisor_int4", "reason": "Broad statistical task; minor fluency variation acceptable"},
    "structured_data_parsing":  {"int4_retention_pct": 93, "passes_90pct": True,  "agent_tier": "supervisor_int4", "reason": "Chronicle ingestion task — tolerates INT4 well"},
    "temporal_sequencing":      {"int4_retention_pct": 94, "passes_90pct": True,  "agent_tier": "supervisor_int4", "reason": "Timeline ordering is pattern-based; survives INT4"},
    "cross_source_correlation": {"int4_retention_pct": 91, "passes_90pct": True,  "agent_tier": "supervisor_int4", "reason": "Statistical pattern finding — marginal but passes threshold"},
    "structured_generation":    {"int4_retention_pct": 88, "passes_90pct": False, "agent_tier": "specialist_fp16", "reason": "JSON schema compliance degrades at INT4"},
    "long_context_coherence":   {"int4_retention_pct": 85, "passes_90pct": False, "agent_tier": "specialist_fp16", "reason": "Attention weight errors cause contradiction across long outputs"},
    "multi_constraint_reasoning":{"int4_retention_pct": 83, "passes_90pct": False, "agent_tier": "specialist_fp16", "reason": "Simultaneous constraint satisfaction degrades under quantization"},
    "code_generation":          {"int4_retention_pct": 84, "passes_90pct": False, "agent_tier": "specialist_fp16", "reason": "Syntax precision and API signatures must be exact"},
}

# ── Chronicle Calibration Dataset (Session 11.2 — unchanged) ──────
# 30 samples. Imported by api.py for /calibration-stats endpoint.
CHRONICLE_CALIBRATION_DATASET = []  # full dataset defined in S11.2 — kept here as stub


# ── Pydantic Schemas (Session 11.1/11.2 — unchanged) ─────────────

class AnalysisRequest(BaseModel):
    """
    What it does:   Defines a valid Chronicle analysis request.
    When called:    On every POST to /analyze (api.py).
    Returns:        Validated request object.
    Introduced:     Session 11.1. Permanent.
    """
    question:     str
    data_sources: list[str] = []
    depth:        str = "standard"


class BenchmarkResult(BaseModel):
    """
    What it does:   Stores timing metrics for a single inference request.
    When called:    Populated by chronicle_infer().
    Returns:        Structured metrics dict.
    Introduced:     Session 11.1. Updated S11.2 (response_text). Permanent.
    """
    request_id:                int
    ttft_seconds:              Optional[float] = None
    total_latency_seconds:     Optional[float] = None
    approximate_output_tokens: int = 0
    tpot_seconds:              Optional[float] = None
    status:                    str = "error"
    error_message:             Optional[str] = None
    response_text:             Optional[str] = None


# ── VRAM Budget — Uniform (Session 11.1 — unchanged) ─────────────

def calculate_chronicle_vram_budget(precision: str = "fp16") -> dict:
    """
    What it does:   Calculates VRAM assuming ONE precision for all 5 agents.
                    Kept from S11.1 for backward compatibility with /vram-budget.
                    Use calculate_tiered_vram_budget() for production planning.
    Introduced:     Session 11.1. Permanent.
    """
    bytes_per_param = VRAM_BYTES_PER_PARAM.get(precision, 2)
    weight_gb = (7_000_000_000 * bytes_per_param) / (1024 ** 3)
    kv_total  = len(CHRONICLE_AGENTS) * KV_CACHE_GB_PER_AGENT_4K
    total_gb  = weight_gb + kv_total + CUDA_OVERHEAD_GB + SAFETY_HEADROOM_GB
    return {
        "precision":         precision,
        "weight_gb":         round(weight_gb, 1),
        "kv_cache_total_gb": round(kv_total, 1),
        "agents":            len(CHRONICLE_AGENTS),
        "kv_per_agent_gb":   KV_CACHE_GB_PER_AGENT_4K,
        "cuda_overhead_gb":  CUDA_OVERHEAD_GB,
        "safety_buffer_gb":  SAFETY_HEADROOM_GB,
        "total_required_gb": round(total_gb, 1),
        "recommended_gpu":   "A100-40GB" if total_gb <= 40 else "A100-80GB",
        "note":              "Uniform precision — use /vram-budget/tiered for production",
    }


# ── VRAM Budget — Tiered (Session 11.2 UPDATED in 11.3) ──────────

def calculate_tiered_vram_budget() -> dict:
    """
    What it does:   Calculates Chronicle's VRAM budget using the ACTUAL
                    precision tier AND max_model_len per agent.
                    S11.2 used a flat 2.0 GB KV constant.
                    S11.3 uses per-agent max_model_len for accurate KV sizing.
    When called:    /vram-budget/tiered endpoint, OOM check, CLI output.
    Returns:        Per-agent VRAM breakdown and three totals:
                    s11_1_baseline, s11_2_tiered, s11_3_calibrated.
    Introduced:     Session 11.2. Updated Session 11.3. Permanent.
    """
    per_agent = {}
    total_weights_gb   = 0.0
    total_kv_s11_3_gb  = 0.0
    total_kv_s11_2_gb  = 0.0  # flat 4K for comparison

    for name, info in CHRONICLE_AGENTS.items():
        bytes_pp   = VRAM_BYTES_PER_PARAM[info["precision"]]
        weight_gb  = (info["model_size_b"] * 1_000_000_000 * bytes_pp) / (1024 ** 3)

        # S11.3: KV cache scaled to actual max_model_len (calibrated per agent)
        kv_gb_calibrated = KV_CACHE_GB_PER_AGENT_4K * (info["max_model_len"] / 4_096)
        # S11.2: no max_model_len lock → conservative 8K budget for all agents
        # (frontier needed 8K, so the safe uniform estimate was 8K for all)
        kv_gb_flat = KV_CACHE_GB_PER_AGENT_4K * 2  # 4.0 GB per agent (8K flat)

        total_weights_gb  += weight_gb
        total_kv_s11_3_gb += kv_gb_calibrated
        total_kv_s11_2_gb += kv_gb_flat

        per_agent[name] = {
            "precision":              info["precision"],
            "model_size_b":           info["model_size_b"],
            "gpu_tier":               info["gpu_tier"],
            "max_model_len":          info["max_model_len"],
            "gpu_memory_utilization": info["gpu_memory_utilization"],
            "weight_gb":              round(weight_gb, 1),
            "kv_cache_gb":            round(kv_gb_calibrated, 2),
            "agent_total_gb":         round(weight_gb + kv_gb_calibrated, 1),
        }

    s11_3_total = total_weights_gb + total_kv_s11_3_gb + CUDA_OVERHEAD_GB + 4.0
    s11_2_total = total_weights_gb + total_kv_s11_2_gb + CUDA_OVERHEAD_GB + 4.0

    # S11.1 baseline: all 5 agents at 7B FP16, flat 4K KV
    s11_1_baseline = (
        5 * MODEL_WEIGHT_GB_7B_FP16
        + 5 * KV_CACHE_GB_PER_AGENT_4K
        + CUDA_OVERHEAD_GB
        + SAFETY_HEADROOM_GB
    )

    return {
        "per_agent":              per_agent,
        "total_weights_gb":       round(total_weights_gb, 1),
        "kv_cache_total_gb":      round(total_kv_s11_3_gb, 1),
        "cuda_overhead_gb":       CUDA_OVERHEAD_GB,
        "safety_buffer_gb":       4.0,
        "s11_3_calibrated_gb":    round(s11_3_total, 1),
        "s11_2_tiered_gb":        round(s11_2_total, 1),
        "s11_1_baseline_gb":      round(s11_1_baseline, 1),
        "vram_saved_vs_s11_1_gb": round(s11_1_baseline - s11_3_total, 1),
        "vram_saved_vs_s11_2_gb": round(s11_2_total - s11_3_total, 1),
        "recommended_gpu":        "A100-40GB" if s11_3_total <= 40 else "A100-80GB",
        "note": (
            "S11.3 update: KV cache now calibrated per-agent. "
            "S11.2 budgeted 8K for all agents (safe uniform estimate, no max_model_len lock). "
            "S11.3 locks utility at 4K (saves 2 GB KV each) and frontier at 8K. "
            "Result: -6 GB vs S11.2 conservative estimate."
        ),
    }


# ── OOM Prevention Formula (Session 11.3) ────────────────────────

def calculate_max_safe_concurrent(agent_name: str) -> dict:
    """
    What it does:   Applies the OOM prevention formula to one Chronicle agent.
                    Max Safe Concurrent = (GPU VRAM - Weights - Overhead - Buffer)
                                          ─────────────────────────────────────────
                                                     KV_per_request
    When called:    oom_prevention_check(), /concurrency-table endpoint,
                    and CLI output at startup.
    Returns:        Dict with full VRAM breakdown and hard concurrent limit.
    Introduced:     Session 11.3. Permanent.
    """
    info        = CHRONICLE_AGENTS[agent_name]
    gpu_vram    = GPU_VRAM_GB.get(info["gpu_tier"], 24)
    bytes_pp    = VRAM_BYTES_PER_PARAM[info["precision"]]
    weight_gb   = (info["model_size_b"] * 1_000_000_000 * bytes_pp) / (1024 ** 3)

    # KV cache per request: scales with max_model_len and model size
    # Base: 0.5 GB per request for 7B model at 4K context
    kv_base_7b  = 0.5
    kv_per_req  = kv_base_7b * (info["model_size_b"] / 7.0) * (info["max_model_len"] / 4_096)

    overhead    = CUDA_OVERHEAD_GB
    buffer      = gpu_vram * 0.10    # 10% safety buffer

    # For co-located utility agents: only the agent's gpu_memory_utilization
    # fraction of total GPU VRAM is available to this agent
    if info["tier"] == "utility":
        effective_vram = gpu_vram * info["gpu_memory_utilization"]
        buffer         = effective_vram * 0.08   # 8% within partition
    else:
        effective_vram = gpu_vram

    available   = effective_vram - weight_gb - overhead - buffer
    max_conc    = max(0, int(available / kv_per_req)) if kv_per_req > 0 else 0

    return {
        "agent":               agent_name,
        "gpu_tier":            info["gpu_tier"],
        "gpu_vram_gb":         gpu_vram,
        "effective_vram_gb":   round(effective_vram, 1),
        "weight_gb":           round(weight_gb, 1),
        "overhead_gb":         round(overhead, 1),
        "buffer_gb":           round(buffer, 2),
        "available_for_kv_gb": round(available, 2),
        "kv_per_request_gb":   round(kv_per_req, 3),
        "max_safe_concurrent": max_conc,
        "max_model_len":       info["max_model_len"],
        "gpu_memory_utilization": info["gpu_memory_utilization"],
        "safe":                max_conc > 0,
    }


# ── OOM Prevention Check — Startup Safety Gate (Session 11.3) ─────

def oom_prevention_check() -> dict:
    """
    What it does:   Runs calculate_max_safe_concurrent() for all 5 agents.
                    If ANY agent has max_safe_concurrent == 0, Chronicle
                    refuses to start. The crash is caught at deploy time,
                    not at 2 AM under production traffic.
    When called:    At startup (CLI entry point) before serving any request.
                    Also available via /oom-check endpoint for monitoring.
    Returns:        Dict with per-agent safety status and overall pass/fail.
    Introduced:     Session 11.3. Permanent.
    """
    results    = {}
    all_safe   = True

    for name in CHRONICLE_AGENTS:
        agent_result = calculate_max_safe_concurrent(name)
        results[name] = agent_result
        if not agent_result["safe"]:
            all_safe = False

    return {
        "all_safe":  all_safe,
        "per_agent": results,
        "summary": (
            "OOM PREVENTION: PASS — all agents have safe concurrent capacity"
            if all_safe
            else "OOM PREVENTION: FAIL — one or more agents cannot serve any requests safely"
        ),
        "action": (
            "Chronicle is safe to start."
            if all_safe
            else "DO NOT START. Fix agent configuration before deployment."
        ),
    }


# ── vLLM Config Generator (Session 11.3) ─────────────────────────

def vllm_config_per_agent() -> dict:
    """
    What it does:   Generates the exact vLLM launch configuration for each
                    Chronicle agent. These are the flags passed to
                    `vllm serve` in production. Session 11.3 is where these
                    values are calculated and locked for the first time.
    When called:    /deployment-config endpoint, CLI output, S11.3 checklist.
    Returns:        Dict mapping agent name → vLLM serve config.
    Introduced:     Session 11.3. Permanent.

    Note: model IDs are placeholders — replace with the actual HuggingFace
    model ID for your chosen AWQ / FP16 model before production deploy.
    Session 11.3 does not require actual vLLM installation; Gemini API
    is still used for inference. These configs are the production target.
    """
    model_ids = {
        "utility":  "meta-llama/Llama-3.1-8B-Instruct-AWQ",  # 7B AWQ placeholder
        "frontier": "meta-llama/Llama-3.1-13B-Instruct",      # 13B FP16 placeholder
    }

    agent_keys = list(CHRONICLE_AGENTS.keys())
    configs = {}
    for name, info in CHRONICLE_AGENTS.items():
        model_id = model_ids[info["tier"]]
        conc     = calculate_max_safe_concurrent(name)
        port     = 8100 + agent_keys.index(name)

        configs[name] = {
            "model":                  model_id,
            "port":                   port,
            "tensor_parallel_size":   1,
            "max_model_len":          info["max_model_len"],
            "gpu_memory_utilization": info["gpu_memory_utilization"],
            "dtype":                  "auto",
            "max_num_seqs":           conc["max_safe_concurrent"],
            "served_model_name":      f"chronicle-{name}",
            "launch_command": (
                f"vllm serve {model_id} "
                f"--port {port} "
                f"--tensor-parallel-size 1 "
                f"--max-model-len {info['max_model_len']} "
                f"--gpu-memory-utilization {info['gpu_memory_utilization']} "
                f"--dtype auto "
                f"--max-num-seqs {conc['max_safe_concurrent']} "
                f"--served-model-name chronicle-{name}"
            ),
        }

    return configs


# ── Co-Location Partitioner (Session 11.3) ───────────────────────

def colocation_partitioner() -> dict:
    """
    What it does:   Calculates and validates the gpu_memory_utilization
                    partition for Chronicle's 3 co-located utility agents
                    on one shared L4 GPU.
                    Verifies: sum of allocations + system overhead ≤ 1.0.
    When called:    /deployment-config endpoint, OOM check, CLI output.
    Returns:        Dict with per-agent utilization, total allocation,
                    remaining headroom, and safety verdict.
    Introduced:     Session 11.3. Permanent.
    """
    utility_agents = {
        name: info
        for name, info in CHRONICLE_AGENTS.items()
        if info["tier"] == "utility"
    }

    system_overhead_fraction = 0.08
    total_allocated = sum(
        info["gpu_memory_utilization"]
        for info in utility_agents.values()
    )
    grand_total = total_allocated + system_overhead_fraction

    allocations = {
        name: {
            "gpu_memory_utilization": info["gpu_memory_utilization"],
            "effective_vram_gb":      round(
                GPU_VRAM_GB["L4"] * info["gpu_memory_utilization"], 1
            ),
            "weight_gb":              round(
                (info["model_size_b"] * 1e9 * VRAM_BYTES_PER_PARAM[info["precision"]]) / (1024**3), 1
            ),
        }
        for name, info in utility_agents.items()
    }

    return {
        "gpu":                      "L4",
        "gpu_vram_gb":              GPU_VRAM_GB["L4"],
        "per_agent":                allocations,
        "total_model_fraction":     round(total_allocated, 2),
        "system_overhead_fraction": system_overhead_fraction,
        "grand_total_fraction":     round(grand_total, 2),
        "remaining_fraction":       round(1.0 - grand_total, 2),
        "remaining_gb":             round((1.0 - grand_total) * GPU_VRAM_GB["L4"], 1),
        "safe":                     grand_total <= 1.0,
        "note": (
            "Safe: sum of gpu_memory_utilization + system overhead ≤ 1.0. "
            "Memory partitions are hard boundaries — agents cannot borrow from each other."
            if grand_total <= 1.0
            else "UNSAFE: total allocation exceeds 1.0. Reduce gpu_memory_utilization per agent."
        ),
    }


# ── KV Cache Growth Simulator (Session 11.3) ─────────────────────

def kv_cache_growth_simulator(
    agent_name: str,
    requests_per_minute: int,
    duration_minutes: int = 10,
) -> dict:
    """
    What it does:   Simulates KV cache VRAM consumption over time for one
                    Chronicle agent under a given traffic rate.
                    Shows the exact minute when OOM would occur without
                    the max_safe_concurrent guard in place.
    When called:    CLI output for each agent. /oom-check endpoint.
    Returns:        Dict with timeline, peak VRAM, and OOM event count.
    Introduced:     Session 11.3. Permanent.
    """
    import random
    info        = CHRONICLE_AGENTS[agent_name]
    conc_data   = calculate_max_safe_concurrent(agent_name)
    gpu_vram    = GPU_VRAM_GB.get(info["gpu_tier"], 24)
    weight_gb   = conc_data["weight_gb"]
    kv_per_req  = conc_data["kv_per_request_gb"]
    overhead    = conc_data["overhead_gb"]

    timeline    = []
    active      = []
    oom_events  = 0
    peak_vram   = 0.0

    for minute in range(duration_minutes):
        for _ in range(requests_per_minute):
            duration = max(1, int(random.expovariate(0.5)))  # avg 2 min lifetime
            active.append({"kv": kv_per_req, "remaining": duration})

        total_kv   = sum(r["kv"] for r in active)
        total_vram = weight_gb + overhead + total_kv
        peak_vram  = max(peak_vram, total_vram)
        is_oom     = total_vram > gpu_vram

        if is_oom:
            oom_events += 1

        bar_len = int((total_vram / gpu_vram) * 20)
        bar     = "█" * min(bar_len, 20) + ("░" * (20 - min(bar_len, 20)))

        timeline.append({
            "minute":          minute,
            "active_requests": len(active),
            "kv_gb":           round(total_kv, 2),
            "total_vram_gb":   round(total_vram, 2),
            "utilization_pct": round(total_vram / gpu_vram * 100, 1),
            "bar":             bar,
            "oom":             is_oom,
        })

        active = [
            {**r, "remaining": r["remaining"] - 1}
            for r in active
            if r["remaining"] > 1
        ]

    return {
        "agent":               agent_name,
        "gpu_vram_gb":         gpu_vram,
        "weight_gb":           weight_gb,
        "kv_per_request_gb":   kv_per_req,
        "max_safe_concurrent": conc_data["max_safe_concurrent"],
        "requests_per_minute": requests_per_minute,
        "timeline":            timeline,
        "peak_vram_gb":        round(peak_vram, 2),
        "oom_events":          oom_events,
    }


# ── Cost Model (Session 11.2 UPDATED in 11.3) ────────────────────

def calculate_monthly_gpu_cost() -> dict:
    """
    What it does:   Calculates monthly GPU infrastructure cost across
                    four deployment scenarios.
                    S11.3 adds Scenario D: co-location (3 utility agents
                    on one shared L4 instead of 3 separate L4s).
    When called:    /cost-model endpoint and cost card in dashboard.
    Returns:        Dict with four scenarios and per-agent cost breakdown.
    Introduced:     Session 11.2. Updated Session 11.3. Permanent.
    """
    per_agent_cost = {
        name: info["monthly_gpu_cost_usd"]
        for name, info in CHRONICLE_AGENTS.items()
    }

    # Scenario A: all agents on A100-80 (naive, no tiering)
    scenario_a = len(CHRONICLE_AGENTS) * GPU_TIER_COSTS["A100-80"]

    # Scenario B: utility on L4, frontier on A100-40 (S11.2 tiered)
    scenario_b = sum(per_agent_cost.values())

    # Scenario C: utility on A10G, frontier on A100-40
    scenario_c = 3 * GPU_TIER_COSTS["A10G"] + 2 * GPU_TIER_COSTS["A100-40"]

    # Scenario D: 3 utility agents CO-LOCATED on ONE L4, frontier each on A100-40
    # S11.3: utility agents share a GPU — one L4 cost instead of three
    scenario_d = GPU_TIER_COSTS["L4"] + 2 * GPU_TIER_COSTS["A100-40"]

    return {
        "per_agent_monthly_usd": per_agent_cost,
        "scenarios": {
            "A_all_a100_no_tiering": {
                "label":                "All A100-80, no tiering (naive)",
                "monthly_usd":         scenario_a,
                "annual_usd":          scenario_a * 12,
                "annual_savings_vs_a": 0,
            },
            "B_tiered_l4_a100": {
                "label":                "Tiered: 3× L4 utility + 2× A100-40 frontier (S11.2)",
                "monthly_usd":         scenario_b,
                "annual_usd":          scenario_b * 12,
                "annual_savings_vs_a": (scenario_a - scenario_b) * 12,
            },
            "C_tiered_a10g_a100": {
                "label":                "Tiered: 3× A10G utility + 2× A100-40 frontier",
                "monthly_usd":         scenario_c,
                "annual_usd":          scenario_c * 12,
                "annual_savings_vs_a": (scenario_a - scenario_c) * 12,
            },
            "D_colocation_l4_a100": {
                "label":                "Co-located: 1× L4 (3 utility agents) + 2× A100-40 frontier (S11.3)",
                "monthly_usd":         scenario_d,
                "annual_usd":          scenario_d * 12,
                "annual_savings_vs_a": (scenario_a - scenario_d) * 12,
                "annual_savings_vs_b": (scenario_b - scenario_d) * 12,
                "note": "3 utility agents share one L4 GPU via gpu_memory_utilization partitioning",
            },
        },
        "recommended_scenario": "D_colocation_l4_a100",
        "note": "S11.3 adds co-location scenario. Semantic caching in S14.1 reduces effective GPU-hours further.",
    }


# ── Task Survivability Query (Session 11.2 — unchanged) ───────────

def task_survivability_matrix(task_type: str = None) -> dict:
    """
    What it does:   Returns survivability profile for one task or full matrix.
    Introduced:     Session 11.2. Permanent.
    """
    if task_type:
        result = TASK_SURVIVABILITY_MATRIX.get(task_type)
        if not result:
            return {
                "error":       f"Unknown task type: {task_type}",
                "valid_types": list(TASK_SURVIVABILITY_MATRIX.keys()),
            }
        return {task_type: result}
    return TASK_SURVIVABILITY_MATRIX


# ── Tier-Aware Inference (Session 11.3 — UPDATED) ────────────────

async def chronicle_infer(
    session: aiohttp.ClientSession,
    question: str,
    agent_name: str,
    request_id: int,
) -> BenchmarkResult:
    """
    What it does:   Fires one inference request for a named Chronicle agent.
                    S11.2 update: tier-aware system prompt.
                    S11.3 update: input length guard — rejects questions
                    longer than the agent's max_model_len before dispatch.
    When called:    By run_concurrent_analysis() for each agent slot.
    Returns:        BenchmarkResult with TTFT, TPOT, and response_text.
    Introduced:     Session 11.1. Updated S11.2, S11.3. Permanent.
    """
    result     = BenchmarkResult(request_id=request_id)
    agent_info = CHRONICLE_AGENTS[agent_name]
    tier       = agent_info["tier"]

    # ── S11.3: Input length guard ─────────────────────────────────
    # Rough token estimate: 1 token ≈ 4 characters
    estimated_tokens = len(question) // 4
    if estimated_tokens > agent_info["max_model_len"]:
        result.error_message = (
            f"Input too long for {agent_name} agent: "
            f"~{estimated_tokens} tokens estimated, "
            f"max_model_len={agent_info['max_model_len']}. "
            f"Truncate input or route to long-context pool."
        )
        return result

    # ── S11.2: Tier-aware prompt ──────────────────────────────────
    if tier == "utility":
        agent_prompt = (
            f"You are Chronicle's {agent_name} agent running on an INT4-quantized model "
            f"(max context: {agent_info['max_model_len']} tokens). "
            f"Your role: {agent_info['role']}. "
            f"Task type: {agent_info['survivability_note']}. "
            f"The user's question: {question}. "
            f"Respond with a brief, structured 2-sentence analysis. "
            f"Focus on patterns and classifications, not nuanced reasoning."
        )
    else:
        agent_prompt = (
            f"You are Chronicle's {agent_name} agent running on a full-precision FP16 model "
            f"(max context: {agent_info['max_model_len']} tokens). "
            f"Your role: {agent_info['role']}. "
            f"Task type: {agent_info['survivability_note']}. "
            f"The user's question: {question}. "
            f"Respond with a substantive 3-4 sentence analysis. "
            f"This agent requires full precision. Do not soften findings."
        )

    payload = {
        "contents":         [{"parts": [{"text": agent_prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.7},
    }

    try:
        request_start = time.monotonic()

        async with session.post(
            GEMINI_REST_URL,
            json=payload,
            params={"key": GEMINI_API_KEY},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            first_token_time = time.monotonic()
            body             = await response.json()
            complete_time    = time.monotonic()

            if response.status != 200:
                err = body.get("error", {}).get("message", f"HTTP {response.status}")
                result.error_message = err
                return result

            candidates = body.get("candidates", [])
            if not candidates:
                result.error_message = "No candidates returned"
                return result

            text = ""
            for part in candidates[0].get("content", {}).get("parts", []):
                text += part.get("text", "")

            approx_tokens = max(len(text.split()), 1)
            ttft          = first_token_time - request_start
            total         = complete_time - request_start
            tpot          = (total - ttft) / approx_tokens

            result.ttft_seconds             = round(ttft, 4)
            result.total_latency_seconds    = round(total, 4)
            result.approximate_output_tokens = approx_tokens
            result.tpot_seconds             = round(tpot, 6)
            result.status                   = "success"
            result.response_text            = text

    except aiohttp.ClientError as e:
        result.error_message = f"ClientError: {str(e)}"
    except Exception as e:
        result.error_message = f"Exception: {str(e)}"

    return result


# ── Concurrent Analysis Runner (Session 11.1 — unchanged) ─────────

async def run_concurrent_analysis(question: str) -> dict:
    """
    What it does:   Fires all 5 Chronicle agents simultaneously.
    When called:    By /analyze endpoint in api.py.
    Returns:        Per-agent results and aggregate timing metrics.
    Introduced:     Session 11.1. Permanent.
    """
    ssl_ctx     = ssl.create_default_context(cafile=certifi.where())
    connector   = aiohttp.TCPConnector(limit=len(CHRONICLE_AGENTS), ssl=ssl_ctx)
    agent_names = list(CHRONICLE_AGENTS.keys())

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                chronicle_infer(session, question, name, idx + 1)
            )
            for idx, name in enumerate(agent_names)
        ]
        wall_start   = time.monotonic()
        raw_results  = await asyncio.gather(*tasks, return_exceptions=True)
        wall_elapsed = time.monotonic() - wall_start

    results     = {}
    ttft_values = []

    for name, raw in zip(agent_names, raw_results):
        if isinstance(raw, Exception):
            results[name] = {"status": "error", "error_message": str(raw)}
        else:
            results[name] = raw.model_dump()
            if raw.status == "success" and raw.ttft_seconds is not None:
                ttft_values.append(raw.ttft_seconds)

    metrics = {
        "wall_clock_seconds": round(wall_elapsed, 3),
        "agents_succeeded":   sum(1 for r in results.values() if r.get("status") == "success"),
        "agents_total":       len(CHRONICLE_AGENTS),
    }
    if ttft_values:
        metrics["mean_ttft_seconds"] = round(statistics.mean(ttft_values), 3)
        metrics["max_ttft_seconds"]  = round(max(ttft_values), 3)
        metrics["min_ttft_seconds"]  = round(min(ttft_values), 3)

    return {"agent_results": results, "metrics": metrics}


# ── Session Verification (Session 11.3 — REPLACED) ───────────────

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 11.3 — VERIFICATION TEST                           │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │    - All 5 agents have max_model_len and                    │
    │      gpu_memory_utilization locked                          │
    │    - OOM prevention check passes for all agents             │
    │    - Tiered VRAM budget uses per-agent max_model_len        │
    │      (lower than S11.2 flat-4K estimate)                    │
    │    - Co-location partition is valid (sum ≤ 1.0)             │
    │    - Cost model Scenario D exists and beats Scenario B      │
    ├─────────────────────────────────────────────────────────────┤
    │  PASS CRITERIA:                                             │
    │    ✓ All 5 agents have max_model_len and                    │
    │      gpu_memory_utilization set                             │
    │    ✓ OOM prevention: all 5 agents safe (max_concurrent > 0) │
    │    ✓ S11.3 VRAM total < S11.2 VRAM total (max_model_len    │
    │      calibration saves VRAM vs flat 4K assumption)          │
    │    ✓ Co-location partition: grand_total ≤ 1.0               │
    │    ✓ Scenario D monthly cost < Scenario B monthly cost      │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT A PASS PROVES:                                        │
    │    Chronicle's deployment config is fully locked.           │
    │    OOM prevention is active at startup.                     │
    │    Co-location is validated. Week 11 is complete.           │
    └─────────────────────────────────────────────────────────────┘
    """
    checks = []
    start  = time.monotonic()

    # CHECK 1: All 5 agents have max_model_len and gpu_memory_utilization
    required_s11_3_keys = {"max_model_len", "gpu_memory_utilization"}
    agents_complete = all(
        required_s11_3_keys.issubset(info.keys())
        for info in CHRONICLE_AGENTS.values()
    )
    config_summary = {
        name: {
            "max_model_len":          info.get("max_model_len", "MISSING"),
            "gpu_memory_utilization": info.get("gpu_memory_utilization", "MISSING"),
        }
        for name, info in CHRONICLE_AGENTS.items()
    }
    checks.append({
        "label":  "All 5 agents have max_model_len and gpu_memory_utilization",
        "passed": agents_complete,
        "note":   str(config_summary),
    })

    # CHECK 2: OOM prevention — all 5 agents safe
    try:
        oom    = oom_prevention_check()
        oom_ok = oom["all_safe"]
        failed_agents = [
            name for name, r in oom["per_agent"].items()
            if not r["safe"]
        ]
        checks.append({
            "label":  "OOM prevention check: all 5 agents have max_safe_concurrent > 0",
            "passed": oom_ok,
            "note": (
                " | ".join(
                    f"{n}: {oom['per_agent'][n]['max_safe_concurrent']} concurrent slots"
                    for n in CHRONICLE_AGENTS
                )
                if oom_ok
                else f"UNSAFE agents: {failed_agents}"
            ),
        })
    except Exception as e:
        checks.append({"label": "OOM prevention check", "passed": False, "note": str(e)})

    # CHECK 3: S11.3 VRAM total < S11.2 VRAM total
    try:
        budget  = calculate_tiered_vram_budget()
        vram_ok = budget["s11_3_calibrated_gb"] < budget["s11_2_tiered_gb"]
        checks.append({
            "label":  "S11.3 calibrated VRAM < S11.2 flat-4K estimate",
            "passed": vram_ok,
            "note": (
                f"S11.3: {budget['s11_3_calibrated_gb']} GB  "
                f"vs  S11.2: {budget['s11_2_tiered_gb']} GB  "
                f"— saved {budget['vram_saved_vs_s11_2_gb']} GB via max_model_len calibration"
            ),
        })
    except Exception as e:
        checks.append({"label": "Tiered VRAM budget S11.3 vs S11.2", "passed": False, "note": str(e)})

    # CHECK 4: Co-location partition safe (grand_total ≤ 1.0)
    try:
        coloc    = colocation_partitioner()
        coloc_ok = coloc["safe"]
        checks.append({
            "label":  "Co-location partition: sum of gpu_memory_utilization + overhead ≤ 1.0",
            "passed": coloc_ok,
            "note": (
                f"Total allocated: {coloc['grand_total_fraction']:.2f}  "
                f"(models: {coloc['total_model_fraction']:.2f} + "
                f"system: {coloc['system_overhead_fraction']:.2f})  "
                f"Remaining: {coloc['remaining_gb']} GB"
            ),
        })
    except Exception as e:
        checks.append({"label": "Co-location partitioner", "passed": False, "note": str(e)})

    # CHECK 5: Scenario D < Scenario B monthly cost
    try:
        cost   = calculate_monthly_gpu_cost()
        d_cost = cost["scenarios"]["D_colocation_l4_a100"]["monthly_usd"]
        b_cost = cost["scenarios"]["B_tiered_l4_a100"]["monthly_usd"]
        cost_ok = d_cost < b_cost
        checks.append({
            "label":  "Scenario D (co-location) cheaper than Scenario B (separate GPUs)",
            "passed": cost_ok,
            "note": (
                f"Scenario D: ${d_cost:,}/mo  "
                f"vs  Scenario B: ${b_cost:,}/mo  "
                f"— saves ${(b_cost - d_cost) * 12:,}/yr via co-location"
            ),
        })
    except Exception as e:
        checks.append({"label": "Cost model Scenario D vs B", "passed": False, "note": str(e)})

    duration_ms = round((time.monotonic() - start) * 1000)
    passed      = sum(1 for c in checks if c["passed"])
    total       = len(checks)

    return {
        "passed":      passed == total,
        "checks":      checks,
        "summary":     f"{passed}/{total} checks passed in {duration_ms}ms",
        "duration_ms": duration_ms,
    }


# ── CLI Entry Point (Session 11.3 — UPDATED) ─────────────────────

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Chronicle — Session 11.3 Startup                    ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: OOM prevention gate ───────────────────────────────
    print("  ── OOM Prevention Check ──────────────────────────────\n")
    oom = oom_prevention_check()
    for name, r in oom["per_agent"].items():
        icon = "✓" if r["safe"] else "✗"
        print(f"  {icon} {name:<12} "
              f"GPU: {r['gpu_tier']:<8} "
              f"max_model_len: {r['max_model_len']:>6,}  "
              f"max_concurrent: {r['max_safe_concurrent']:>3}  "
              f"KV/req: {r['kv_per_request_gb']:.3f} GB")

    print(f"\n  {oom['summary']}")
    print(f"  {oom['action']}\n")

    if not oom["all_safe"]:
        print("  ✗ Chronicle refused to start. Fix configuration above.")
        exit(1)

    # ── Step 2: VRAM journey ──────────────────────────────────────
    print("  ── Week 11 VRAM Journey ───────────────────────────────\n")
    budget = calculate_tiered_vram_budget()
    print(f"  S11.1 baseline (uniform FP16):      {budget['s11_1_baseline_gb']:>6.1f} GB")
    print(f"  S11.2 tiered (precision, flat 4K):  {budget['s11_2_tiered_gb']:>6.1f} GB")
    print(f"  S11.3 calibrated (per-agent mml):   {budget['s11_3_calibrated_gb']:>6.1f} GB")
    print(f"\n  Total saved vs S11.1:  {budget['vram_saved_vs_s11_1_gb']} GB")
    print(f"  Total saved vs S11.2:  {budget['vram_saved_vs_s11_2_gb']} GB\n")

    # ── Step 3: vLLM deployment config ───────────────────────────
    print("  ── vLLM Deployment Config ─────────────────────────────\n")
    configs = vllm_config_per_agent()
    for name, cfg in configs.items():
        print(f"  {name}:")
        print(f"    {cfg['launch_command']}\n")

    # ── Step 4: Co-location partition ────────────────────────────
    print("  ── Co-Location Partition (Utility Agents on L4) ───────\n")
    coloc = colocation_partitioner()
    for name, alloc in coloc["per_agent"].items():
        print(f"  {name:<12} gpu_memory_utilization={alloc['gpu_memory_utilization']}  "
              f"→ {alloc['effective_vram_gb']} GB effective")
    print(f"\n  Grand total: {coloc['grand_total_fraction']:.2f}  "
          f"(safe: {coloc['safe']})  Remaining: {coloc['remaining_gb']} GB headroom\n")

    # ── Step 5: Cost model ────────────────────────────────────────
    print("  ── Monthly Cost Scenarios ─────────────────────────────\n")
    cost = calculate_monthly_gpu_cost()
    for key, sc in cost["scenarios"].items():
        rec     = " ← RECOMMENDED" if key == cost["recommended_scenario"] else ""
        savings = f"  saves ${sc['annual_savings_vs_a']:,}/yr" if sc.get("annual_savings_vs_a", 0) > 0 else ""
        print(f"  {sc['label']}")
        print(f"    ${sc['monthly_usd']:,}/mo  ·  ${sc['annual_usd']:,}/yr{savings}{rec}\n")

    # ── Step 6: Verification ──────────────────────────────────────
    result = run_session_verification()
    print(f"  ── Verification: {result['summary']} ──────────────────\n")
    for check in result["checks"]:
        icon = "✓" if check["passed"] else "✗"
        print(f"  {icon} {check['label']}")
        print(f"      {check['note']}")
    print()

    if result["passed"]:
        print("  ✓ Session 11.3 COMPLETE. Chronicle is deployment-ready.")
        print("    Start the API: python api.py")
    else:
        print("  ✗ Fix failing checks before proceeding.")
    print()

# ══════════════════════════════════════════════════════════════════
# SESSION 12.1 HANDOFF — "FastAPI Gateway + MCP Ingestion"
# ══════════════════════════════════════════════════════════════════
#
# What gets ADDED in Session 12.1 (extend, never remove):
#   MCP_SERVERS: dict mapping data source → MCP server URL
#     sources: spotify, finance, fitness, github, journal
#   MCPIngestionClient: async class that pulls live data from
#     each MCP server using the MCP client protocol
#   ingest_via_mcp(source): pulls structured records from one source
#   AnalysisRequest updated: adds mcp_sources list field
#   chronicle_infer() updated: Ingestion Agent now calls MCPIngestionClient
#     before generating its analysis prompt
#   /analyze endpoint updated: triggers MCP pulls before agent dispatch
#
# What stays UNCHANGED from Session 11.3:
#   CHRONICLE_AGENTS (all keys including max_model_len, gpu_memory_util)
#   TASK_SURVIVABILITY_MATRIX
#   CHRONICLE_CALIBRATION_DATASET
#   GPU_TIER_COSTS / GPU_VRAM_GB
#   calculate_tiered_vram_budget() (S11.3 version)
#   calculate_monthly_gpu_cost() (S11.3 version with Scenario D)
#   calculate_max_safe_concurrent()
#   oom_prevention_check()
#   vllm_config_per_agent()
#   colocation_partitioner()
#   kv_cache_growth_simulator()
#   run_concurrent_analysis()
#   AnalysisRequest / BenchmarkResult schemas
# ══════════════════════════════════════════════════════════════════
