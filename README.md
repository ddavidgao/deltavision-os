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

- [x] OSWorld platform + runner wired ([`capture/osworld.py`](capture/osworld.py), [`benchmarks/run_osworld.py`](benchmarks/run_osworld.py)). Real API (`task_config` dict, PNG-bytes obs, pyautogui-string actions). Loads `test_small.json` (39 tasks) cleanly.
- [x] **OSWorld VM running**: native Docker in WSL2 Ubuntu on the Windows 5080 box (bypassing Docker Desktop's credential store quirks). VM container exposes port 5000 (Flask), 8006 (VNC), 9222 (Chrome DevTools). Full `env.reset → env.step('WAIT') → env.evaluate()` round-trip verified on a Chrome task.
- [~] **Partial Phase 3 (UI-TARS Q4)**: A/B on 6 overlapping clean-run OSWorld tasks. DeltaVision-ON used **40.8% fewer tokens** (23.2k vs 39.2k) but UI-TARS terminated significantly earlier (4.2 vs 10.3 avg steps). Force-full-frame solved 1 task (GIMP, 26 steps); DeltaVision-ON solved 0.

- [x] **Phase 3 diagnostic (Sonnet 4.6 via Claude Code subagent, GIMP task)**: rules out the "model-agnostic bug" hypothesis. Sonnet with **delta** observations: 15 steps, score 0.0, **did not early-terminate** (handled delta crops by reading menu-text fragments). Sonnet with **full frames**: 16 steps, score **1.0** (succeeded). Combined with the UI-TARS result, the picture sharpens:
    - *DeltaVision is not broken.* A general reasoning model accepts delta observations and keeps acting.
    - *UI-TARS' early-termination is model-specific.* Action-tuned agents may misinterpret delta crops as "task done" due to how they were trained.
    - *Full frames still help spatial recovery.* When a click misses its target, full-screen context each step lets the model see "where am I" and self-correct. Delta crops limit this.
    - The paper framing shifts from "model-agnostic middleware" to **"DeltaVision saves tokens, with a recovery-context trade-off that's model-dependent; pair it with a model that can either handle delta crops or receive full frames on ambiguity."**

- [~] **A11y-hybrid observation layer** (WIP — matches V1's v1.0.2 DOM+focus unlock, ported to OS). Parser + pruner + schema lives in [`observation/a11y.py`](observation/a11y.py), 18 passing unit tests in [`tests/test_a11y.py`](tests/test_a11y.py). Novelty: uses the pixel-diff bbox as the gate for which a11y nodes enter the prompt (UFO2 filters by interactivity; OSWorld filters by role whitelist; DeltaVision-OS filters by what *changed* + what's focused). Next: wire into `agent/loop.py` + model backends + test against a live OSWorld task.

- [ ] Full OSWorld A/B across ≥20 tasks on a Sonnet-class model (current blocker: subagent-driven runs take ~30 min/task × 2 modes × N tasks ≫ practical. Needs Anthropic API key or equivalent direct inference path).
- [ ] smart_resize-aware client preprocessing to close UI-TARS' ~25pp gap to published FP16 numbers
- [ ] Migration of V1 paper section 5 (OS-level experiments)

### OSWorld setup notes (WSL2 Ubuntu 24.04)

The clean path on Windows is WSL2 native docker, not Docker Desktop (which silently fails from detached SSH sessions due to the `docker-credential-desktop.exe` credential store needing an interactive Windows session).

```bash
# In WSL2 Ubuntu
git clone https://github.com/xlang-ai/OSWorld.git  # or use /mnt/c/...
python3 -m venv oswo_venv && source oswo_venv/bin/activate
pip install -U setuptools wheel                     # numpy~=1.24 pin in requirements.txt breaks on py3.12
pip install numpy pillow gymnasium pyautogui requests requests-toolbelt flask \
    psutil filelock tqdm pandas opencv-python-headless fabric playwright docker \
    pydrive PyDrive2 rapidfuzz lxml cssselect tldextract formulas pytz pyyaml \
    func-timeout imagehash pymupdf pdfplumber striprtf chardet mutagen ebooklib \
    pyperclip pypdf pyacoustid music-tag lark fastdtw easyocr ag2 scikit-image \
    scikit-learn librosa

# Clear Docker config that points at Windows-side cred helper
echo {} > ~/.docker/config.json

# First DesktopEnv construction downloads Ubuntu.qcow2.zip (~11GB) and extracts (~24GB)
python3 -c "from desktop_env.desktop_env import DesktopEnv; \
  env = DesktopEnv(provider_name='docker', os_type='Ubuntu', \
                   action_space='pyautogui', require_a11y_tree=False); \
  print('ok')"
```

## License

MIT. Same as V1.
