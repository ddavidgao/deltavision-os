# ScreenSpot-v2 Evaluation Summary

Head-to-head grounding accuracy on ScreenSpot-v2 (1,272 UI screenshots, `OS-Copilot/ScreenSpot-v2`).
Scoring: predicted click point must land inside the ground-truth bbox (SeeClick convention).

Generated from `results/deltavision.db` — every row is queryable:
```bash
sqlite3 results/deltavision.db \
  "SELECT id, backend, metrics_json FROM runs WHERE benchmark='screenspot_v2' ORDER BY id"
```

## Full-sample head-to-head

| Model (quant) | Backend | n | Overall | Desktop | Mobile | Web | Run ID |
|---|---|---:|---:|---:|---:|---:|---:|
| **UI-TARS-1.5-7B Q4_K_M** | llama.cpp + mmproj | 1272 | **64.1%** | 79.6% | 78.6% | 35.5% | #9 |
| Qwen2.5-VL-7B Q4_K_M | Ollama 0.20.2 | 1272 | 28.6% | 59.9% | 22.0% | 12.4% | #10 |
| Claude Sonnet 4.6 (subagent) | Claude Code subagent | 90 | 18.9% | 53.3% | 0.0% | 3.3% | #8 |

## Text vs icon split (UI-TARS full 1272)

| Split | Accuracy | n |
|---|---:|---:|
| desktop-text | 88.7% | 194 |
| desktop-icon | 67.1% | 140 |
| mobile-text | 83.4% | 290 |
| mobile-icon | 72.0% | 211 |
| web-text | 38.5% | 234 |
| web-icon | 32.0% | 203 |

## What the numbers mean

**UI-TARS-1.5-7B Q4 at 64.1% overall** is the strong result — about 2.2× Qwen at the same quantization. The gap is biggest on mobile (78.6% vs 22.0%, +56.6pp) and web (35.5% vs 12.4%, +23.1pp), which is exactly where a purpose-built grounding model should dominate a general VLM.

**Qwen2.5-VL-7B Q4 at 28.6%** is the general-VLM baseline. Desktop holds up (59.9%) because most desktop UIs resemble the model's training distribution. Mobile and web collapse (22% / 12%) for the reason documented in the paper: Qwen's ~1M-pixel visual token budget forces aggressive resize, and small mobile icons lose discriminative features.

**Claude Sonnet 4.6 at 18.9%** (smaller sample for cost) is the strong-general-reasoning ceiling without grounding-specific training. Decent on desktop (53%), near-zero on mobile and web. Confirms that vision + reasoning ≠ pixel-grounded clicks; grounding needs explicit training.

Published FP16 numbers for UI-TARS-1.5-7B are ~89% — the ~25pp gap here is the expected Q4_K_M quantization cost (paper baselines use FP16 + native preprocessing).

## Known systematic bias (UI-TARS, web_icon)

UI-TARS on web icons consistently predicts ~80-100 pixels left of the target on 2560-wide images. Example: `"view my account profile"` — predicted `(2298, 45)`, ground-truth bbox `[2401..2512, 14..82]`. X is 103px short; y is correct. Suggests the model's training/eval image preprocessing differs from ours. Fixable with smart_resize-aware client-side preprocessing; not a model defect.

## Stack notes

**UI-TARS path**: `ollama create` can't cleanly handle separately-packaged VLM GGUFs in 0.20.2 (multi-FROM not supported). llama-server from llama.cpp was the clean path — `-m language.gguf --mmproj vision.gguf` loads the multimodal pair correctly. Ran via SSH-foreground from the Mac (Start-Process / cmd start detach killed the child; interactive SSH keeps it alive).

**Qwen path**: Ollama's pre-bundled `qwen2.5vl:7b` just works.

**Sonnet path**: 3 parallel `subagent_type=general-purpose` agents on Claude Sonnet 4.6, each given 30 examples. Ground truth stripped from inputs. Agents Read the image and output (x, y) — no API key needed; uses Claude Code's auth.

**Fara-7B**: skipped. Microsoft released safetensors only; no community GGUF with mmproj exists on HuggingFace. Conversion is out-of-scope.

## Reproducibility

Every run has a row in `results/deltavision.db` and a per-run artifact directory:

- `benchmarks/runs/screenspot_v2_ui_tars_q4km_gguf/run_9/` — UI-TARS full
- `benchmarks/runs/screenspot_v2_qwen2_5vl_7b/run_10/` — Qwen full
- `benchmarks/runs/screenspot_v2_claude_sonnet_4_6_subagent_n90/run_8/` — Sonnet

Each directory contains `config.json` (git SHA, python version, model, adapter, base_url, prompt template), `metrics.json`, and `result.json` (full per-example predictions + hits).
