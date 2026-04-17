# DeltaVision-OS

Delta-first computer-use agent framework for **OS-level** and **OSWorld** environments.

Sibling to [`deltavision`](https://github.com/ddavidgao/deltavision), which targets browsers via Playwright. This project extends the DeltaVision observation middleware to the full desktop: any native application, any OS task, any OSWorld VM benchmark.

## Scope

| | `deltavision` (V1) | `deltavision-os` (V2, this repo) |
|---|---|---|
| Observation source | Playwright screenshots | `mss` OS-level capture, OSWorld VM frames |
| Action space | click, type, scroll, key, wait | + drag, double-click, right-click, hotkey |
| Eval targets | Wikipedia, TodoMVC, GitHub, classifier sites | OSWorld 369-task suite |
| Model backends | Claude, OpenAI, Ollama | + llama.cpp server (MAI-UI-8B, Qwen3-VL-8B) |
| Dependencies | Playwright + 5 pip packages | + mss, pyautogui, OSWorld harness |
| Status | Frozen @ paper artifact | Active development |

**If you want browser automation, use V1.** If you want desktop / OS / OSWorld, use this.

## Status

Under construction. Architecture:

```
capture/   # Platform abstraction (OS, OSWorld, VNC)
execute/   # Action executor abstraction
vision/    # CV pipeline (diff, classify, pHash) — ported from V1
agent/     # Loop, state, typed actions
model/     # VLM backends + llama.cpp
eval/      # OSWorld eval harness integration
```

## Shared concept with V1

The core insight is identical: a zero-LLM CV pipeline gates what the model sees. Full frame on `NEW_PAGE`, delta thumbnail + crops on `DELTA`. Same 4-layer classifier cascade. Same ~80% token savings on sticky-context tasks.

The **platform abstraction** is new. V1 had three Playwright-specific callsites in its loop; V2 replaces them with a generic `Platform` interface that any capture+execute backend can implement.

## License

MIT. Same as V1.
