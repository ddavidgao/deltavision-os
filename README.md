# DeltaVision-OS

Delta-first computer-use agent framework for **OS-level** and **OSWorld** environments.

Sibling to [`deltavision`](https://github.com/ddavidgao/deltavision), which targets browsers via Playwright. This project extends the DeltaVision observation middleware to the full desktop: any native application, any OS task, any OSWorld VM benchmark.

## Status

- **238 tests** passing (229 offline, 9 need a real display)
- **Live V2 E2E**: real Qwen2.5-VL on an RTX 5080 via SSH tunnel, driving a real Mac desktop through the agent loop — 5 steps, 4.6% median diff, 56% hypothetical token savings vs full-frame. Video at [`benchmarks/v2_live_demo.mp4`](benchmarks/v2_live_demo.mp4).
- **Matched-trajectory ablation** ([`benchmarks/run_ablation_os.py`](benchmarks/run_ablation_os.py)): DeltaVision-gated vs forced full-frame on the same 10-step desktop script — **68.2% token savings**. Sensitivity sweep across 3 trajectories × 3 NEW_PAGE_DIFF_THRESHOLD values shows savings is threshold-insensitive in [0.30, 0.75] (pHash layer dominates).
- **ScreenSpot-v2 grounding head-to-head** ([`benchmarks/screenspot_summary.md`](benchmarks/screenspot_summary.md)):

  | Model (Q4) | n | Overall | Desktop | Mobile | Web |
  |---|---:|---:|---:|---:|---:|
  | **UI-TARS-1.5-7B** | 1272 | **64.1%** | 79.6% | 78.6% | 35.5% |
  | Qwen2.5-VL-7B | 1272 | 28.6% | 59.9% | 22.0% | 12.4% |
  | Claude Sonnet 4.6 | 90 | 18.9% | 53.3% | 0% | 3.3% |

  Purpose-built grounding models (UI-TARS) beat general VLMs (Qwen) beat strong general reasoning (Claude) at pixel-level click prediction — 2-3× factor on the same quant budget.

- **Reproducibility**: every benchmark result writes a row to `results/deltavision.db` with auto-increment run ID + config snapshot (git SHA, python version, model, endpoint). Per-run artifact directories under `benchmarks/runs/{bench}_{backend}/run_{id}/` preserve the full result JSON. Same pattern as V1.

- OSWorld integration still stubbed.

## Scope

| | `deltavision` (V1) | `deltavision-os` (V2, this repo) |
|---|---|---|
| Observation source | Playwright screenshots | `mss` OS-level capture, OSWorld VM frames |
| Action space | click, type, scroll, key, wait | + drag, double-click, right-click, hotkey |
| Eval targets | Wikipedia, TodoMVC, GitHub, classifier sites | OSWorld 369-task suite |
| Model backends | Claude, OpenAI, Ollama | + llama.cpp / OpenAI-compat server (Qwen2.5-VL verified, MAI-UI-8B / Qwen3-VL targeted) |
| Dependencies | Playwright + 5 pip packages | + mss, pyautogui, OSWorld harness |
| Status | Frozen @ paper artifact | Active development |

**If you want browser automation, use V1.** If you want desktop / OS / OSWorld, use this.

## Quick Start

**From PyPI** (library only):

```bash
pip install deltavision-os
```

**From source** (recommended for running the benchmarks — dataset downloads, harness scripts, and tests only ship in the repo):

```bash
git clone https://github.com/ddavidgao/deltavision-os.git
cd deltavision-os
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 238 tests (9 need a real display; those skip on CI)
pytest tests/ -q

# Live desktop benchmark (pure CV, no model needed)
python benchmarks/desktop_idle_observe.py --rounds 5 --interval 0.5
```

See [TESTS.md](TESTS.md) for a full breakdown of what each test covers.

Expected benchmark output on a quiet desktop:

```
step   1  delta     diff=0.000  phash= 0  anchor=1.00  trigger=none
step   2  delta     diff=0.000  phash= 0  anchor=1.00  trigger=none
step   3  delta     diff=0.000  phash= 0  anchor=1.00  trigger=none
step   4  delta     diff=0.000  phash= 0  anchor=1.00  trigger=none
step   5  delta     diff=0.000  phash= 0  anchor=1.00  trigger=none

Observed 5 steps in 3.1s
DELTA:     5 (100.0%)
NEW_PAGE:  0

Token savings if paired with a VLM at 1600 tok/full_frame, ~400 tok/delta:
  Full frame every step:  8,000 tokens
  DeltaVision gated:      2,000 tokens (6,000 saved)
```

### Running the CLI

```bash
# Scripted model: real capture, no real actions (safe demo)
python main.py --task "observe desktop" --platform os --backend scripted --max-steps 5

# Claude API (real model, but DRIVES YOUR MOUSE — start with small max-steps)
export ANTHROPIC_API_KEY=sk-...
python main.py --task "..." --platform os --backend claude --safety strict --max-steps 10

# Local VLM over an OpenAI-compatible endpoint (llama.cpp / vLLM / SGLang / Ollama via tunnel)
python main.py --task "..." --platform os --backend llamacpp \
    --host 127.0.0.1 --port 11434 --model qwen2.5vl:7b

# Ablation: force full-frame (disable delta gating)
python main.py --task "..." --platform os --backend claude --force-full-frame
```

**Warning:** `--platform os` drives the REAL mouse and keyboard. Start with `--safety strict` and small `--max-steps` until you trust the model.

## Architecture

```
deltavision-os/
├── capture/          # Platform abstraction (5-method ABC)
│   ├── base.py           Platform class
│   ├── os_native.py      mss + pyautogui impl (macOS/Linux/Windows)
│   └── osworld.py        OSWorld VM wrapper (stub)
├── vision/           # Zero-LLM CV pipeline (ported from V1)
│   ├── diff.py, classifier.py, phash.py, crops.py
├── agent/
│   ├── loop.py           Platform-agnostic agent loop
│   ├── state.py          Observation + response history
│   └── actions.py        10 typed actions (V1's 6 + DRAG/DOUBLE_CLICK/RIGHT_CLICK/HOTKEY)
├── observation/      # FullFrame + Delta observation types
├── model/            # Pluggable backends
│   ├── base.py, _response_parser.py  shared
│   ├── llamacpp.py       V2 new: OpenAI-compat for local VLMs
│   ├── scripted.py       for testing without API costs
│   └── claude.py, openai.py, ollama.py  (carried from V1)
├── safety.py         # Model-agnostic action validation
├── config.py         # All thresholds, validated at construction
├── results/          # SQLite result store
├── benchmarks/       # desktop_idle_observe, pipeline_perf, classifier_sensitivity, record_live_demo
├── main.py           # CLI entrypoint
└── tests/            # 238 passing (229 offline, 9 need display)
```

## Shared concept with V1

The core insight is identical: a zero-LLM CV pipeline gates what the model sees. Full frame on `NEW_PAGE`, delta thumbnail + crops on `DELTA`. Same 4-layer classifier cascade. Same ~80% token savings on sticky-context tasks.

The **platform abstraction** is new. V1 had three Playwright-specific callsites in its loop; V2 replaces them with a generic `Platform` interface that any capture+execute backend can implement.

## Reproducibility

Every benchmark run produces an auto-increment row in SQLite + a per-run artifact directory:

```bash
# list every result row
sqlite3 results/deltavision.db \
  "SELECT id, benchmark, backend, timestamp, notes FROM runs ORDER BY id"

# inspect one run's config snapshot + metrics
cat benchmarks/runs/screenspot_v2_ui_tars_q4km_gguf/run_9/config.json
cat benchmarks/runs/screenspot_v2_ui_tars_q4km_gguf/run_9/metrics.json
```

The config snapshot captures git SHA, python version, platform, model name, adapter, API base URL, prompt template, and any benchmark-specific parameters (thresholds, trajectories, etc.). This mirrors V1's pattern — every published number has a queryable row behind it.

Helper is in [`benchmarks/_repro.py`](benchmarks/_repro.py); all benchmark scripts go through `save_run(...)` to write both the row and the artifact dir in one shot.

## What's working

- [x] Platform ABC with async context manager lifecycle
- [x] OSNativePlatform: mss capture + pyautogui actions (macOS verified)
- [x] OSWorld platform stub (waits for env harness)
- [x] 4-layer CV classifier cascade (URL → diff → pHash → anchor) ported from V1
- [x] Agent loop with force-refresh on no-effect streaks
- [x] 10 action types including DRAG with x2/y2
- [x] Safety layer (credential / URL / action limits)
- [x] Model backends: Claude, OpenAI, Ollama, llama.cpp server, scripted
- [x] 238 passing tests
- [x] Live desktop benchmark proves CV pipeline works without browser
- [x] **First real V2 E2E**: Qwen2.5-VL on remote RTX 5080 via SSH tunnel, 5-step Mac-desktop run with 56% hypothetical token savings ([`benchmarks/v2_live_demo.mp4`](benchmarks/v2_live_demo.mp4))
- [x] Classifier sensitivity sweep (synthetic damage 0%→99%) confirms pHash is the first layer to fire on real transitions
- [x] **Matched-trajectory ablation**: 68.2% token savings vs forced full-frame, robust across NEW_PAGE_DIFF_THRESHOLD ∈ [0.30, 0.75]
- [x] **ScreenSpot-v2 head-to-head**: UI-TARS-1.5-7B Q4 (64.1%) vs Qwen2.5-VL-7B Q4 (28.6%) vs Claude Sonnet 4.6 (18.9%) on n=1272
- [x] Reproducibility discipline: SQLite DB + artifact dirs + config snapshots for every benchmark run

## What's next

- [ ] OSWorld VM integration (needs OSWorld env install)
- [ ] smart_resize-aware client preprocessing to close UI-TARS' ~25pp gap to published FP16 numbers
- [ ] Full end-to-end agent run on an OS task (current ScreenSpot measures grounding in isolation; need to pair with a real task trajectory)
- [ ] Migration of V1 paper section 5 (OS-level experiments)

## License

MIT. Same as V1.
