# DeltaVision-OS Test Coverage

Visual map of what every test verifies. **267 tests** total, **258 pass offline** (9 need a real display, skipped on CI).

```
Total: 267 tests
├── CV pipeline         26 tests  (diff + phash + classifier)
├── Agent layer         69 tests  (state + loop + actions)
├── Model response      33 tests  (JSON extraction, VLM quirks)
├── Observation         12 tests  (builder + post_init invariants)
├── A11y hybrid         18 tests  (parser + diff-bbox pruner + focused element)
├── Naive-install       11 tests  (subprocess install/import/shadow invariants)
├── Safety layer        37 tests  (URL, credentials, action limits, presets)
├── Config validation   45 tests  (every field, every bound)
├── Results store       19 tests  (SQLite persistence)
├── Platform (mss)       7 tests  (needs display - skipped on CI)
├── Real-capture E2E     2 tests  (needs display - skipped on CI)
```

## 1. Agent layer — 69 tests

### `tests/test_actions.py` — 38 tests (V2-new)

V2 extends V1's 6-action space to 10. Every parsing path needs coverage.

| Test class | Covers |
|---|---|
| `TestNativeFormat` (10) | All 10 ActionType values parse correctly from DeltaVision-native JSON |
| `TestUITarsFormat` (7) | UI-TARS / CogAgent format mapping (left_click→CLICK, finished→DONE, unknown→None) |
| `TestCoercion` (6) | String coords become ints (MAI-UI-8B quirk), float truncation, drag endpoints, invalid strings |
| `TestEdgeCases` (5) | None / empty dict / string input / unknown type / missing coords |
| `TestStringRepresentations` (10) | `__str__` doesn't crash for any action type — parametrized |

### `tests/test_state.py` — 13 tests (V2-new)

`AgentState` dataclass. The agent's memory of what it's seen and done.

| Test class | Covers |
|---|---|
| `TestInitialState` (2) | default values, empty delta_ratio |
| `TestNoChangeStreak` (3) | increment / reset / reset-from-zero |
| `TestNewPageCount` (1) | increment |
| `TestTransitionLog` (2) | basic log entry, multiple transitions |
| `TestDeltaRatio` (3) | all-delta=1.0, all-new-page=0.0, mixed=0.75 |
| `TestObservationResponseLogging` (2) | observations and responses append correctly |

### `tests/test_v2_loop_scripted.py` — 8 tests (V2-new)

End-to-end integration: MockPlatform + ScriptedModel + agent loop.

| Test | What it proves |
|---|---|
| `test_empty_script_terminates_after_initial_capture` | loop exits cleanly when model returns done |
| `test_single_click_action_triggers_classification` | 1 action → 1 executed → 1 transition logged |
| `test_no_change_streak_forces_full_refresh` | stuck actions → `force_refresh_no_effect` trigger |
| `test_platform_get_url_none_does_not_crash_loop` | **V2-critical:** OS-native returns URL=None, classifier must handle |
| `test_drag_action_type_roundtrips` | V2 DRAG action with x2/y2 parses, executes, logs |
| `test_new_page_via_diff_ratio_increments_counter` | Full pixel swap → NEW_PAGE → new_page_count++ |
| `test_safety_block_increments_step_without_executing` | blocked action never reaches platform.execute() |
| `test_force_full_frame_mode` | FORCE_FULL_FRAME=True makes every obs full_frame |

## 2. CV pipeline — 26 tests

Ported from V1 unchanged — this is the core classifier logic.

| File | Tests | Covers |
|---|---|---|
| `test_diff.py` | 8 | identical frames, full swap, small change, noise filtering, MAX_REGIONS cap, crop bounds |
| `test_phash.py` | 4 | identical → 0, different → high, similar → low, hash size = 64 |
| `test_classifier.py` | 14 | All 4 cascade layers: URL (2), diff ratio (1), anchor match (2), scroll bypass (5), animation guard (2), extract_anchor (2) |

## 3. Observation — 12 tests

### `tests/test_observation_builder.py` — 12 tests (V2-new)

| Test class | Covers |
|---|---|
| `TestFullFrameBuilder` (3) | basic shape, last_action, default url/trigger |
| `TestDeltaBuilder` (4) | basic shape, empty crops default, text_deltas, current_frame |
| `TestDispatch` (1) | unknown obs_type falls through to delta |
| `TestPostInitInvariants` (2) | obs_type always normalized via __post_init__ |
| *(inherited)* | base `Observation` shape covered via subclass tests |

## 4. Model response parsing — 33 tests

Identical to V1. Critical defense against local-VLM output quirks.

| Test class | What breaks if this fails |
|---|---|
| `TestPureJSON` (3) | Well-formed JSON no longer parses |
| `TestCodeFences` (4) | ```json fences break backends (Qwen-VL habit) |
| `TestBraceExtraction` (3) | Prose preamble / postamble breaks parsing |
| `TestFallback` (5) | Malformed output crashes loop instead of stopping |
| `TestNormalizeConfidenceHoisting` (3) | MAI-UI-8B confidence-in-action resurfaces |
| `TestNormalizeAltDoneFields` (5) | `finish` / `finished` / `complete` not recognized as done |
| `TestNormalizeDefaults` (2) | Missing fields cause KeyError downstream |
| `TestGetConfidence` (8) | Numeric strings, garbage, out-of-range confidence |

## 5. Safety layer — 37 tests

Identical to V1 (including the shortener flag bug fix).

| Test class | Defense surface |
|---|---|
| `TestURLShorteners` (5) | bit.ly / tinyurl / t.co blocked; flag disables |
| `TestSuspiciousPatterns` (3) | `.ru`, `password-reset`, `account-verify` URLs |
| `TestDomainAllowlist` (4) | allowlist enforced, None means permissive |
| `TestCredentialDetection` (8) | SSN / credit card / CVV detection |
| `TestSensitiveFieldContext` (3) | typing into `password` / `ssn` fields blocked |
| `TestActionLimits` (5) | oversized type, negative coords, None coords |
| `TestPresets` (4) | PERMISSIVE / STRICT / EDUCATIONAL distinct behaviors |
| `TestNonTypeActions` (3) | click / scroll / key skip credential check |
| `TestSafetyResult` (2) | dataclass defaults |

## 6. Config — 45 tests

Identical to V1. Every threshold validated at construction.

| Test class | Validator |
|---|---|
| `TestDefaults` (2) | defaults construct, MCGRAWHILL preset |
| `TestFractions` (15) | 7 fraction fields × boundaries |
| `TestPHashThresholds` (4) | pHash in [0, 64], animation margin |
| `TestPositiveInts` (14) | 10 int fields non-negative, floats rejected |
| `TestQuantization` (8) | None / "4bit" / "8bit" only |
| `TestAnchorBBox` (6) | length=4, x2>x1, y2>y1, non-degenerate |

## 7. Results store — 19 tests

Identical to V1. SQLite schema + persistence.

| Test class | Covers |
|---|---|
| `TestSave` (9) | ID sequencing, flattened fields, JSON blobs, legacy names, NULL |
| `TestQuery` (3) | list of dicts, param binding, empty result |
| `TestBest` (4) | lowest-time, best_ms over avg_ms, metrics rehydrated |
| `TestSchema` (3) | tables + indexes, reopen preserves data |

## 8. Platform — 7 tests (display-required)

### `tests/test_os_native_capture.py` — 7 tests

Real `mss` capture on Mac. **Skipped on CI** — no display.

| Test class | Covers |
|---|---|
| `TestCapture` (4) | returns PIL Image, sane dimensions, RGB mode, two captures mostly identical |
| `TestURL` (1) | get_url returns None (OS has no URL concept) |
| `TestLifecycle` (2) | capture-before-setup raises, teardown clears mss |

## 9. Real-capture E2E — 2 tests (display-required)

### `tests/test_v2_real_capture.py`

Hybrid tests: real mss capture + scripted model + full agent loop.

| Test | What it proves |
|---|---|
| `test_pipeline_runs_against_real_frames` | full loop completes on real Mac captures without crash |
| `test_quiet_desktop_classifies_as_delta` | quiet desktop → DELTA, not NEW_PAGE (diff < 30%) |

## How to run

```bash
# Full suite (needs display)
pytest tests/ -q
# → 238 passed in ~13s

# CI-equivalent (skip display-requiring tests)
pytest tests/ -q \
    --ignore=tests/test_os_native_capture.py \
    --ignore=tests/test_v2_real_capture.py
# → 229 passed in ~0.5s

# Single module
pytest tests/test_safety.py -v

# By test class
pytest tests/test_v2_loop_scripted.py::TestLoopIntegration -v

# With coverage (needs pytest-cov)
pytest tests/ --cov=. --cov-report=term-missing
```

## Known gaps (deliberately not tested)

- **Real mouse / keyboard execution via pyautogui** — no test drives the actual cursor. Adding one would require consistent accessibility permissions across CI runners. Covered by manual integration (e.g., `main.py --platform os --backend scripted --max-steps N`).
- **OSWorld end-to-end** — platform stub only; full coverage deferred until OSWorld env is installed.
- **Real llama.cpp server round-trip** — depends on Windows server being up. Covered by live manual runs.
- **Model backends with mocked clients** — `test_response_parser.py` covers pure parsing in isolation; full `predict()` calls with mocked anthropic/openai/requests clients not yet written.
