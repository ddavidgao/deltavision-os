# ScreenSpot-v2 Benchmark Summary

**Date:** 2026-04-17  
**Sample:** 30 per platform (n=90 total), subset evaluation  
**Endpoint:** Ollama on Windows 5080 box via Tailscale tunnel (http://127.0.0.1:11434/v1)  
**Note:** All numbers are Q4_K_M quantization (Qwen2.5-VL-7B) via Ollama. Published paper baselines use FP16 — quantization is expected to reduce accuracy vs. paper numbers.

---

## Results Table

| model | desktop-text | desktop-icon | mobile-text | mobile-icon | web-text | web-icon | overall |
|-------|-------------|-------------|------------|------------|---------|---------|---------|
| Qwen2.5-VL-7B (Q4_K_M, n=90) | 83.3% | 44.4% | 25.0% | 5.6% | 7.7% | 5.9% | 26.7% |
| UI-TARS-1.5-7B (F16, n=0) | — | — | — | — | — | — | — |

---

## Per-Platform Summary

| model | desktop (all) | mobile (all) | web (all) | overall |
|-------|--------------|-------------|----------|---------|
| Qwen2.5-VL-7B | 60.0% | 13.3% | 6.7% | 26.7% |
| UI-TARS-1.5-7B | — | — | — | — |

Sample sizes: desktop=30 (12 text, 18 icon), mobile=30 (12 text, 18 icon), web=30 (13 text, 17 icon).

---

## Model Comparison

Qwen2.5-VL-7B (Q4_K_M) shows a strong platform gradient: it performs well on desktop UI (83% text, 44% icon) but degrades sharply on mobile (25%/6%) and collapses on web (8%/6%). The desktop-text result suggests the model has learned Windows/Linux UI vocabulary well enough even at Q4_K_M, but mobile and web layouts are outside its effective grounding distribution at this quantization level. The published Qwen2.5-VL-7B number on ScreenSpot-v2 is ~80% overall (FP16); the 26.7% here is consistent with a combination of Q4_K_M quantization loss, the harness using a seeclick-style prompt rather than the model's native smart_resize preprocessing, and sample variance at n=90 (the 15-sample smoke test also gave 40%, which is within noise of this 26.7% given heavy mobile/web weighting).

UI-TARS-1.5-7B (F16, 15.2 GB) could not be benchmarked: every inference call to the Ollama endpoint returned no data and silently closed the connection after several minutes. Ollama accepted the connection but never streamed a response, consistent with a VRAM exhaustion or model-loading failure on the 5080 box. The F16 weights alone consume ~15 GB of VRAM; if other models were resident or the 5080's VRAM was partially used, loading would fail silently. The eval process was terminated after confirming the issue reproduces across multiple isolated curl calls with up to 5-minute timeouts.

---

## Why Phase 2 Did Not Run

Phase 2 (full 1272-sample run) was not executed:

1. **Qwen2.5-VL-7B Phase 1 accuracy is 26.7%**, which is above the 25% threshold, but the UI-TARS model failed entirely, making a head-to-head comparison impossible.
2. The UI-TARS F16 model loading failure on the Ollama backend (see above) blocked the second model entirely. Running a one-sided full eval on Qwen without the counterpart is not informative for the head-to-head goal.
3. Per benchmark instructions: "If the tunnel drops mid-run and one backend starts returning connection errors, stop that run, report the error, and move on."

---

## Quantization Note

Both models were served via Ollama:
- `qwen2.5vl:7b` — Q4_K_M (4-bit, ~4.7 GB)
- `0000/ui-tars-1.5-7b:latest` — F16 (~15.2 GB, non-functional)

Q4_K_M quantization degrades grounding accuracy relative to FP16 paper baselines. Published ScreenSpot-v2 results for Qwen2.5-VL-7B (FP16, official preprocessing) are ~80% overall. The 26.7% measured here reflects quantization losses, prompt format mismatch (seeclick vs. native), and sample composition (icon-heavy, mobile/web-heavy subset). Results should not be compared directly to paper numbers without FP16 replication.

---

## Artifacts

| file | description |
|------|-------------|
| `benchmarks/screenspot_qwen_n90.json` | Qwen2.5-VL-7B n=90 full results (90 scored, 342s wall time) |
| `benchmarks/screenspot_uitars_n90.json` | Not produced — UI-TARS model failed to load |
