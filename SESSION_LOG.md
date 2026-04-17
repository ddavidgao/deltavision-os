# V2 Build Log

## 2026-04-17 (autonomous loop)

### Iteration 1 milestone: V2 runs end-to-end

Starting state: scaffold only (Platform ABC, OS native stub, OSWorld stub).
Ending state of iteration 1: **174 tests passing + live desktop benchmark working.**

### Iteration 2 milestone: V2 is production-shaped

- `main.py` CLI: `--platform os|osworld`, 5 backends (scripted/llamacpp/claude/openai/ollama), safety modes, ablation flag
- `.github/workflows/tests.yml` CI on Python 3.11/3.12/3.13
- README quickstart with real benchmark output
- 3 new loop integration tests: NEW_PAGE, safety block, FORCE_FULL_FRAME (177 total tests)
- All 5 model backends converted to shared `_response_parser` (parallel with V1)
- `pipeline_perf.py` benchmark: measures CV overhead on real Mac captures (41.6ms median — matches V1 paper's claim)

### Cumulative state

**Tests: 177 passing in ~12s**

| Module | Tests | Notes |
|---|---|---|
| `test_classifier.py` | 14 | All 4 cascade layers + scroll bypass |
| `test_config.py` | 45 | Every field validator |
| `test_diff.py` | 8 | Diff computation + bbox extraction |
| `test_os_native_capture.py` | 7 | mss capture sanity on real display |
| `test_phash.py` | 4 | Hamming distance |
| `test_response_parser.py` | 33 | JSON extraction + VLM quirks |
| `test_results_store.py` | 19 | SQLite persistence |
| `test_safety.py` | 37 | Includes V1 shortener bug regression |
| `test_v2_loop_scripted.py` | 8 | end-to-end: empty, single, stuck, url=none, drag, new_page, safety, force_full_frame |
| `test_v2_real_capture.py` | 2 | hybrid real-mss + scripted |

### Live benchmark outputs

```
$ python benchmarks/desktop_idle_observe.py --rounds 5 --interval 0.5
... 5 DELTA classifications, 0 NEW_PAGE, diff=0.000 on quiet desktop
Token savings: 6,000 of 8,000 (75%) if paired with VLM
```

```
$ python benchmarks/pipeline_perf.py --iterations 10
Screen: 1470x956

stage             min      med      p95      max  (ms)
capture          8.5    10.2    16.0    16.0
diff             3.9     4.3    17.1    17.1
classify        23.8    25.6   323.0   323.0
crop             0.0     0.0     0.0     0.0
TOTAL           37.2    41.6   352.8   352.8

At 42ms median, CV adds 4.2% of a 1s inference window.
```

### Full file tree

```
deltavision-os/
├── .github/workflows/tests.yml  # CI: pytest on 3.11/3.12/3.13
├── README.md                     # Scope, quickstart, CLI examples, benchmarks
├── CLAUDE.md                     # Future-session instructions
├── LICENSE, SESSION_LOG.md
├── pyproject.toml                # deltavision-os @ 0.1.0-alpha
├── .gitignore
├── main.py                       # CLI entrypoint
├── agent/
│   ├── __init__.py
│   ├── actions.py                # 10 action types (V1 + DRAG/HOTKEY/etc)
│   ├── loop.py                   # Platform-agnostic
│   └── state.py
├── capture/
│   ├── __init__.py
│   ├── base.py                   # Platform ABC
│   ├── os_native.py              # mss + pyautogui
│   └── osworld.py                # Stub
├── config.py, safety.py
├── vision/                       # CV pipeline (V1 port, platform-agnostic)
│   └── diff/classifier/phash/crops
├── observation/                  # FullFrame + Delta types
├── model/
│   ├── base.py, _response_parser.py (shared)
│   ├── scripted.py, llamacpp.py  (V2 additions)
│   ├── claude.py, openai.py, ollama.py, local.py  (shared with V1)
├── results/store.py              # SQLite
├── benchmarks/
│   ├── desktop_idle_observe.py   # live classifier benchmark
│   └── pipeline_perf.py          # V1-paper-style perf measurement
└── tests/                        # 177 passing
```

### Attempted and parked

- **Remote Ollama via Tailscale** — binds only to 127.0.0.1. SSH port forward connects but Ollama returns empty reply. Parked; needs David's Windows-side config (`OLLAMA_HOST=0.0.0.0` + restart) or llama.cpp install.
- **Public repo sync for CI workflows** — OAuth token for public mirror lacks `workflow` scope. Worked around by adding `.github/` to `.public-exclude`. CI only runs on the private repo.

### Ready for next iteration

- Hook `LlamaCppModel` or `OllamaModel` up to a running server (blocked on VLM endpoint decision).
- Port V1 ablation runner (`run_ablation.py`) to V2 now that CLI + platform + benchmarks are in place.
- OSWorld integration: implement the stub in `capture/osworld.py`.
- Create the GitHub remote for `deltavision-os` and push.

### Iteration 3 additions

- **23 new unit tests:** `test_state.py` (13), `test_observation_builder.py` (10). Both modules were previously only exercised via integration tests; now have dedicated coverage.
- **`TESTS.md`** — visual coverage map with 238 tests broken down by concern (CV, agent, model, observation, safety, config, store, platform).
- **`benchmarks/classifier_sensitivity.py`** — sweeps synthetic damage from 0% to 99% and prints the cascade's response. Real output on a noise-textured 800×600 base frame:

  ```
  damage  diff_ratio  phash  anchor  transition  trigger
  -------  ----------  -----  ------  ----------  -------
    0.0%     0.000       0    1.00   delta       none
    1.0%     0.010       8    1.00   delta       none
    5.0%     0.050      16    1.00   delta       none
   10.0%     0.101      18    1.00   delta       none
   20.0%     0.200      22    1.00   new_page    phash       # Layer 3 fires first
   35.0%     0.350      26    1.00   new_page    phash
   50.0%     0.500      28    1.00   new_page    phash
   75.0%     0.743      20    0.20   new_page    anchor_loss # Block overlaps anchor
   90.0%     0.743      20    0.20   new_page    anchor_loss
   99.0%     0.743      20    0.20   new_page    anchor_loss
  ```

  Key insight: pHash (Layer 3) is the first layer to trigger — it catches 20% damage where diff_ratio is only 0.20 (threshold is 0.75). Diff_ratio alone wouldn't catch the transition until 75%+ change.

### Running totals

- **238 tests** passing in ~13s
- **4 benchmarks**: `desktop_idle_observe`, `pipeline_perf`, `classifier_sensitivity`, + the V1 test_integration simulated run (via ported code)
- **14 commits** on `main` (local only, no remote yet)
- **0 external deps blocking** V2 from shipping — the only "missing pieces" are a VLM endpoint and a GitHub remote, both gated on David's call.
