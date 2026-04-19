# Running the benchmarks

DeltaVision-OS ships three classes of benchmark scripts. Each one writes a
row to `results/deltavision.db` and a per-run artifact directory under
`benchmarks/runs/<bench>_<backend>/run_<id>/` — same reproducibility
discipline as V1.

## Ranked by "cheapest to run first"

| Benchmark | Script | Needs | Time | What it proves |
|---|---|---|---:|---|
| Matched-trajectory ablation | [`run_ablation_os.py`](../benchmarks/run_ablation_os.py) | Real macOS display | ~30 s | 68% token savings, no model |
| Ablation sweep (3 trajectories × 3 thresholds) | [`run_ablation_sweep.py`](../benchmarks/run_ablation_sweep.py) | Real macOS display | ~3 min | Savings robust across NEW_PAGE_DIFF_THRESHOLD ∈ [0.30, 0.75] |
| ScreenSpot-v2 grounding | [`screenspot_eval.py`](../benchmarks/screenspot_eval.py) | VLM endpoint | ~5 min / 90 samples | Grounding accuracy per model |
| OSWorld task completion | [`run_osworld.py`](../benchmarks/run_osworld.py) | OSWorld VM + VLM | ~3–5 min/task | End-to-end task success + token ratio |
| **Comprehensive end-to-end test** (this doc, below) | Runbook | All of the above | ~45 min | Cross-app validation of the full stack |

---

## The comprehensive end-to-end test

Purpose: run **3 OSWorld tasks from 3 different app categories** under
**3 different observation configurations** (pixel-only delta, a11y-hybrid
delta, forced full frame). Nine runs total. Gives you a matrix that's
immune to single-task flakes and isolates what the a11y hybrid contributes.

Tasks (in `OSWorld/evaluation_examples/examples/`):

| Category | Task ID | What the agent does |
|---|---|---|
| `gimp` | `554785e9-4523-4e7a-b8e1-8016f565f56a` | Enhance color vibrancy of a photo |
| `libreoffice_writer` | (pick one from test_small.json — not Calc, Calc's a11y hangs) | Edit a document |
| `vs_code` | (pick one from test_small.json) | Modify a code file |

Configurations:

| ID | DeltaVision gating | a11y hybrid | Baseline meaning |
|---|---|---|---|
| A | ON | OFF | Pixel-only delta (the original V2 behavior) |
| B | ON | ON | The hybrid this repo is built around |
| C | OFF (force full frame) | OFF | Sanity reference — every obs is a full frame |

### Step-by-step (Windows WSL2 + 5080)

Set up infrastructure (do once):

```bash
# 1. SSH into Windows, open WSL Ubuntu
ssh david-computer
wsl -d Ubuntu-24.04

# 2. Make sure the venv + OSWorld repo are in place (see troubleshooting.md
#    for first-time setup). Assume /home/david/oswo_venv exists.
source /home/david/oswo_venv/bin/activate

# 3. Pin the deltavision-os version you want to test.
#    Use PyPI for a naive-user run:
pip install --upgrade deltavision-os
#    Or use the git HEAD for the latest WIP:
#    pip install --upgrade git+https://github.com/ddavidgao/deltavision-os.git

# 4. Start llama-server with a grounding model (UI-TARS-1.5-7B Q4_K_M + mmproj).
#    Do this on the Windows host (NOT in WSL) for GPU access, then expose 8080.
#    From a separate Mac/Linux shell:
#    ssh david-computer 'C:\Users\david\llama.cpp\llama-server.exe \
#        -m C:\Users\david\ui-tars-staging\ui-tars-q4km.gguf \
#        --mmproj C:\Users\david\ui-tars-staging\mmproj-q8.gguf \
#        --host 0.0.0.0 --port 8080 -ngl 99 --ctx-size 16384' > /tmp/llama.log 2>&1 &
```

Run the matrix (in WSL, inside the venv):

```bash
# From WSL — Windows host IP is usually 172.25.160.1 (check with `ip route show default`).
export LLAMA=http://172.25.160.1:8080/v1
export OSWORLD=/mnt/c/Users/david/OSWorld

# Config A — pixel-only delta
python -m deltavision_os.main  # or the runner directly:
python benchmarks/run_osworld.py \
    --oswo-repo $OSWORLD \
    --subset test_small.json --max-tasks 3 \
    --model ui-tars-q4km.gguf --adapter ui-tars --base-url $LLAMA \
    --no-safety \
    --output benchmarks/comprehensive_A_pixel_delta.json

# Config B — a11y hybrid (the new thing)
python benchmarks/run_osworld.py \
    --oswo-repo $OSWORLD \
    --subset test_small.json --max-tasks 3 \
    --model ui-tars-q4km.gguf --adapter ui-tars --base-url $LLAMA \
    --a11y-hybrid \
    --no-safety \
    --output benchmarks/comprehensive_B_a11y.json

# Config C — forced full frame baseline
python benchmarks/run_osworld.py \
    --oswo-repo $OSWORLD \
    --subset test_small.json --max-tasks 3 \
    --model ui-tars-q4km.gguf --adapter ui-tars --base-url $LLAMA \
    --force-full-frame \
    --no-safety \
    --output benchmarks/comprehensive_C_full_frame.json
```

What "good" looks like across the 9 runs:

- **C (full frame)** should have the highest token cost AND the highest
  success rate. It's the capability ceiling for this model + task.
- **A (pixel-only delta)** should have the lowest token cost (~40% less
  than C) AND the lowest success rate — this reproduces the "UI-TARS
  early-terminates on delta crops" finding.
- **B (a11y hybrid)** should sit between — tokens closer to A, success
  rate closer to C. That's the claim we're validating.

### What to record

`results/deltavision.db` gets 9 rows with `benchmark='osworld'` and backends
like `ui_tars_q4km_gguf_pixel_delta` / `_a11y_hybrid` / `_force_full_frame`.
Query:

```bash
sqlite3 results/deltavision.db \
    "SELECT id, backend, json_extract(metrics_json, '$.success_count') AS ok,
     json_extract(metrics_json, '$.total_estimated_tokens') AS tok
     FROM runs WHERE benchmark='osworld' ORDER BY id DESC LIMIT 9"
```

Full artifacts per run (config snapshot, per-task result log, token accounting)
are in `benchmarks/runs/osworld_*/run_<id>/`.

---

## Known caveats

- Skip Chrome tasks until the upstream Playwright-asyncio setup bug is
  fixed (see [troubleshooting.md](troubleshooting.md#chrome-tasks-fail-with-playwright-sync-api-inside-the-asyncio-loop)).
- Skip LibreOffice Calc with `require_a11y_tree=True` until upstream
  issue #185 is resolved.
- llama-server at Q4 will give lower absolute success than FP16 published
  numbers; the DELTA between configs A/B/C is what matters for validating
  DeltaVision's claim.
