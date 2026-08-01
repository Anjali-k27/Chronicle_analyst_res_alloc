"""
╔══════════════════════════════════════════════════════════════════╗
║  CHRONICLE — api.py                                              ║
║  Session 11.3: Deployment Config Endpoints Added                 ║
╚══════════════════════════════════════════════════════════════════╝

Changes in Session 11.3 (additions only — nothing removed):
  - /health updated: session → "11.3", new capabilities unlocked
  - GET /deployment-config: vLLM launch config per agent
  - GET /oom-check: OOM prevention check per agent + co-location
  - GET /concurrency-table: context window vs concurrent capacity table
  - GET /vram-budget/tiered: updated (uses S11.3 max_model_len-aware calc)
  - GET /cost-model: updated (Scenario D co-location now included)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os

from agent import (
    AnalysisRequest,
    run_concurrent_analysis,
    calculate_chronicle_vram_budget,
    calculate_tiered_vram_budget,
    calculate_monthly_gpu_cost,
    calculate_max_safe_concurrent,
    task_survivability_matrix,
    oom_prevention_check,
    vllm_config_per_agent,
    colocation_partitioner,
    kv_cache_growth_simulator,
    CHRONICLE_AGENTS,
    GPU_VRAM_GB,
    TASK_SURVIVABILITY_MATRIX,
    VRAM_BYTES_PER_PARAM,
    KV_CACHE_GB_PER_AGENT_4K,
    CUDA_OVERHEAD_GB,
)

app = FastAPI(
    title="Chronicle API",
    description="Local-first personal AI analyst. Session 11.3 — GPU Allocation.",
    version="11.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
async def health():
    """Session 11.3 update: session → "11.3", deployment config in response."""
    budget = calculate_tiered_vram_budget()
    cost   = calculate_monthly_gpu_cost()
    oom    = oom_prevention_check()
    return {
        "status":   "ok",
        "session":  "11.3",
        "version":  "11.3.0",
        "oom_safe": oom["all_safe"],
        "agents": {
            name: {
                "role":                   info["role"],
                "tier":                   info["tier"],
                "precision":              info["precision"],
                "gpu_tier":               info["gpu_tier"],
                "max_model_len":          info["max_model_len"],
                "gpu_memory_utilization": info["gpu_memory_utilization"],
            }
            for name, info in CHRONICLE_AGENTS.items()
        },
        "vram_summary": {
            "s11_3_calibrated_gb":    budget["s11_3_calibrated_gb"],
            "s11_2_tiered_gb":        budget["s11_2_tiered_gb"],
            "s11_1_baseline_gb":      budget["s11_1_baseline_gb"],
            "vram_saved_vs_s11_1_gb": budget["vram_saved_vs_s11_1_gb"],
            "recommended_gpu":        budget["recommended_gpu"],
        },
        "cost_summary": {
            "recommended_scenario":    cost["recommended_scenario"],
            "monthly_usd":             cost["scenarios"]["D_colocation_l4_a100"]["monthly_usd"],
            "annual_savings_vs_naive": cost["scenarios"]["D_colocation_l4_a100"]["annual_savings_vs_a"],
        },
        "capabilities": [
            "concurrent_5_agent_inference",     # S11.1
            "uniform_vram_budget_calculator",   # S11.1
            "tiered_quantization_assignments",  # S11.2
            "task_survivability_matrix",        # S11.2
            "tiered_vram_budget",               # S11.2
            "monthly_gpu_cost_model",           # S11.2
            "chronicle_calibration_dataset",    # S11.2
            "oom_prevention_check",             # S11.3
            "per_agent_max_model_len",          # S11.3
            "colocation_partitioning",          # S11.3
            "vllm_deployment_config",           # S11.3
            "concurrency_table",                # S11.3
            # "mcp_live_data_ingestion",        # S12.1 — not yet
            # "sse_streaming",                  # S12.2 — not yet
            # "async_job_queue",                # S12.3 — not yet
        ],
    }


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """Unchanged from S11.2. S11.3 adds input length guard inside chronicle_infer()."""
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try:
        result = await run_concurrent_analysis(request.question)
        for name, agent_result in result["agent_results"].items():
            agent_result["precision"]              = CHRONICLE_AGENTS[name]["precision"]
            agent_result["gpu_tier"]               = CHRONICLE_AGENTS[name]["gpu_tier"]
            agent_result["max_model_len"]          = CHRONICLE_AGENTS[name]["max_model_len"]
            agent_result["gpu_memory_utilization"] = CHRONICLE_AGENTS[name]["gpu_memory_utilization"]
        return {
            "question":      request.question,
            "agent_results": result["agent_results"],
            "metrics":       result["metrics"],
            "session":       "11.3",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vram-budget")
async def vram_budget(precision: str = "fp16"):
    """Session 11.1: uniform VRAM budget at a given precision. Unchanged."""
    valid = {"fp32", "fp16", "int8", "int4"}
    if precision not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid precision. Must be one of: {sorted(valid)}")
    return calculate_chronicle_vram_budget(precision)


@app.get("/vram-budget/tiered")
async def vram_budget_tiered():
    """Session 11.2/11.3: per-agent tiered budget. S11.3 uses max_model_len-aware KV calc."""
    return calculate_tiered_vram_budget()


@app.get("/cost-model")
async def cost_model():
    """Session 11.2/11.3: monthly GPU cost. S11.3 adds Scenario D co-location."""
    return calculate_monthly_gpu_cost()


@app.get("/survivability")
async def survivability(task_type: str = None):
    """Session 11.2: task survivability matrix. Unchanged."""
    return task_survivability_matrix(task_type)


@app.get("/deployment-config")
async def deployment_config():
    """
    Session 11.3: vLLM launch configuration per agent.
    Returns the exact --flags to pass to `vllm serve` for each Chronicle agent.
    Includes model ID, port, tensor_parallel_size, max_model_len,
    gpu_memory_utilization, max_num_seqs, and the full launch command.
    """
    return {
        "agents":   vllm_config_per_agent(),
        "colocation": colocation_partitioner(),
        "session":  "11.3",
        "note": (
            "Model IDs are placeholders. Replace with your AWQ/FP16 "
            "HuggingFace model ID before production deploy. "
            "Gemini API is used for inference until Session 11.3 hands-on "
            "is complete and local vLLM is configured."
        ),
    }


@app.get("/oom-check")
async def oom_check():
    """
    Session 11.3: OOM prevention check for all 5 Chronicle agents.
    Returns per-agent max_safe_concurrent calculation and overall pass/fail.
    Use this endpoint in your monitoring stack to verify deployment safety.
    Alert if any agent returns max_safe_concurrent == 0.
    """
    oom   = oom_prevention_check()
    coloc = colocation_partitioner()
    return {
        "oom_prevention": oom,
        "colocation":     coloc,
        "all_safe":       oom["all_safe"] and coloc["safe"],
        "session":        "11.3",
    }


@app.get("/concurrency-table")
async def concurrency_table():
    """
    Session 11.3: max_model_len vs max_concurrent_requests table.
    Shows the concurrency cost of each context window size for each agent's GPU.
    Use this to validate that Chronicle's locked max_model_len values
    provide enough concurrent capacity for expected traffic.
    """
    results     = {}
    mml_options = [1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072]

    for name, info in CHRONICLE_AGENTS.items():
        gpu_vram  = GPU_VRAM_GB.get(info["gpu_tier"], 24)
        bytes_pp  = VRAM_BYTES_PER_PARAM[info["precision"]]
        weight_gb = (info["model_size_b"] * 1e9 * bytes_pp) / (1024 ** 3)

        if info["tier"] == "utility":
            effective_vram = gpu_vram * info["gpu_memory_utilization"]
        else:
            effective_vram = gpu_vram

        table = []
        for mml in mml_options:
            kv_per_req = KV_CACHE_GB_PER_AGENT_4K * (mml / 4_096) * (info["model_size_b"] / 7.0)
            overhead   = CUDA_OVERHEAD_GB
            buffer     = effective_vram * 0.10
            available  = effective_vram - weight_gb - overhead - buffer
            max_conc   = max(0, int(available / kv_per_req)) if kv_per_req > 0 else 0
            table.append({
                "max_model_len":  mml,
                "kv_per_req_gb":  round(kv_per_req, 3),
                "max_concurrent": max_conc,
                "locked":         mml == info["max_model_len"],
            })

        results[name] = {
            "gpu_tier":             info["gpu_tier"],
            "effective_vram_gb":    round(effective_vram, 1),
            "locked_max_model_len": info["max_model_len"],
            "table":                table,
        }

    return {"per_agent": results, "session": "11.3"}


if __name__ == "__main__":
    print("\n  Chronicle API — Session 11.3")
    print("  Starting on http://localhost:8000")
    print("  Swagger UI: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
