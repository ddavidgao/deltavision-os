# V2 Build Log

## 2026-04-17 (autonomous loop iteration 1)

### Milestone: V2 runs end-to-end

Starting state: scaffold only (Platform ABC, OS native stub, OSWorld stub).
Ending state: **174 tests passing + live desktop benchmark working.**

### What was built this session

**Ported from V1 (copied + adapted, not depended on):**
- `vision/` — diff engine, classifier cascade, pHash, crops (dropped the Playwright-specific `capture.py`)
- `observation/` — builder + types
- `agent/state.py` — unchanged, already platform-agnostic
- `config.py`, `safety.py` — unchanged
- `model/` — base, _response_parser, scripted, openai, claude, ollama (for potential use)
- `results/store.py` — SQLite store

**New / V2-specific:**
- `agent/actions.py` — extended `ActionType` with `DOUBLE_CLICK`, `RIGHT_CLICK`, `DRAG` (with x2/y2), `HOTKEY`. Parser handles all new types + UI-TARS format.
- `agent/loop.py` — platform-agnostic rewrite. Signature: `run_agent(task, model, platform, config, safety=None)`. 3 Playwright callsites replaced with `platform.*` methods.
- `capture/os_native.py` — fleshed out with cursor_park, lazy pyautogui import, all 10 action types supported in `execute()`.
- `model/llamacpp.py` — subclass of `OpenAIModel` with Tailscale-friendly host/port. Works against any OpenAI-compatible endpoint.
- `benchmarks/desktop_idle_observe.py` — live proof: captures Mac desktop at 1Hz, runs classifier, reports delta ratio + hypothetical token savings.

### Tests (174 total, all passing in ~8s)

| Module | Tests | Notes |
|---|---|---|
| `test_classifier.py` | 14 | All 4 cascade layers + scroll bypass |
| `test_config.py` | 45 | Every field validator |
| `test_diff.py` | 8 | Diff computation + bbox extraction |
| `test_os_native_capture.py` | 7 | **V2-new:** real mss capture, sane dims, RGB, URL=None, lifecycle |
| `test_phash.py` | 4 | Hamming distance |
| `test_response_parser.py` | 33 | JSON extraction + VLM quirks |
| `test_results_store.py` | 19 | SQLite persistence |
| `test_safety.py` | 37 | Includes the V1 shortener-flag bug fix regression |
| `test_v2_loop_scripted.py` | 5 | **V2-new:** end-to-end with MockPlatform + scripted model. Covers empty script, single action, stuck streak → force refresh, URL=None handling, DRAG action |
| `test_v2_real_capture.py` | 2 | **V2-new:** hybrid real-mss + scripted. Real pipeline on real desktop frames. |

### Live benchmark output

```
$ python benchmarks/desktop_idle_observe.py --rounds 5 --interval 0.5

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

### Attempted and parked

- **Remote Ollama via Tailscale** — Windows Ollama binds only to 127.0.0.1. SSH port forward connected but API returned empty reply (Ollama needs `OLLAMA_HOST=0.0.0.0` for remote, or service restart). Parked; needs David's input or Windows-side config.
- **llama.cpp server setup on Windows** — not installed yet. Existing models (qwen2.5vl:7b, minicpm-v, ui-tars-1.5-7b) are in Ollama, would need GGUF download for llama.cpp.

### Ready for next iteration

- Hook `LlamaCppModel` up to a running server (blocked on either getting Ollama to bind externally OR standing up llama.cpp on Windows).
- Port a real V1 benchmark (e.g., `run_ablation.py`) to V2 once a model endpoint is online.
- OSWorld integration: implement the stub in `capture/osworld.py` when OSWorld env is installed.

### File tree

```
deltavision-os/
├── README.md            # V1 vs V2 scope table
├── CLAUDE.md            # Project instructions for future sessions
├── LICENSE              # MIT
├── SESSION_LOG.md       # This file
├── pyproject.toml
├── .gitignore
├── agent/
│   ├── __init__.py
│   ├── actions.py       # V2: +DRAG, +DOUBLE_CLICK, +RIGHT_CLICK, +HOTKEY
│   ├── loop.py          # Platform-agnostic
│   └── state.py
├── capture/
│   ├── __init__.py
│   ├── base.py          # Platform ABC (5 methods + async context manager)
│   ├── os_native.py     # mss + pyautogui
│   └── osworld.py       # Stub
├── config.py
├── safety.py
├── vision/              # CV pipeline (ported from V1)
│   ├── diff.py
│   ├── classifier.py
│   ├── phash.py
│   └── crops.py
├── observation/
│   ├── __init__.py
│   ├── builder.py
│   └── types.py
├── model/
│   ├── base.py
│   ├── _response_parser.py
│   ├── scripted.py      # For testing without API calls
│   ├── llamacpp.py      # V2 new
│   ├── openai.py
│   ├── claude.py
│   └── ollama.py
├── results/
│   └── store.py
├── benchmarks/
│   └── desktop_idle_observe.py   # V2 live benchmark
└── tests/               # 174 passing
```
