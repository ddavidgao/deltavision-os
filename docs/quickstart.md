# Quickstart — using DeltaVision-OS from Python

This guide walks you from `pip install` to your first DeltaVision-gated
observation, in about 30 lines of code. No VM, no GPU, no OSWorld required.

## 1. Install

```bash
pip install deltavision-os
```

Verify the install (do this from any cwd — a `/tmp` is fine):

```bash
python -c "import deltavision_os; print(deltavision_os.__version__)"
```

If that fails, [file an issue](https://github.com/ddavidgao/deltavision-os/issues)
— the project has a regression-guarded naive-install invariant exactly
for this case ([`tests/test_naive_install.py`](../tests/test_naive_install.py)).

## 2. The core loop

Every DeltaVision-OS agent has three pieces:

1. **A Platform** — captures screens, executes actions, speaks one of:
   - [`OSNativePlatform`](../deltavision_os/capture/os_native.py) — mss + pyautogui
     (macOS / Linux / Windows, needs a real display).
   - [`OSWorldPlatform`](../deltavision_os/capture/osworld.py) — wraps an
     OSWorld `DesktopEnv`, hands observations back as PNG bytes + a11y XML.
2. **A Model** — `ScriptedModel` for deterministic testing, `ClaudeModel` /
   `OpenAIModel` / `LlamaCppModel` for real inference.
3. **The agent loop** — `run_agent(task, model, platform, config)`.

The loop capture → diff → classify → build observation → predict → execute
is hidden inside `run_agent`. You just wire the three pieces.

## 3. A runnable smoke test

This starts `OSNativePlatform`, runs a 3-step scripted trajectory, prints
the per-step CV classifier verdict, and shows estimated token cost. No
model API key needed — `ScriptedModel` is purely deterministic.

```python
import asyncio

from deltavision_os import (
    OSNativePlatform, ScriptedModel,
    Action, ActionType,
    DeltaVisionConfig, run_agent,
)

TRAJECTORY = [
    Action(type=ActionType.WAIT, duration_ms=500),
    Action(type=ActionType.WAIT, duration_ms=500),
    Action(type=ActionType.WAIT, duration_ms=500),
]

async def main():
    platform = OSNativePlatform(cursor_park=None)
    model = ScriptedModel(TRAJECTORY)
    config = DeltaVisionConfig()

    async with platform:
        state = await run_agent("smoke", model, platform, config)

    print(f"Ran {state.step} steps.")
    print(f"Delta ratio: {state.delta_ratio:.1%}")
    for t in state.transition_log:
        print(f"  step {t['step']:2d}  "
              f"{t['transition']:<8s}  diff={t['diff_ratio']:.3f}  "
              f"trigger={t['trigger']}")

asyncio.run(main())
```

Expected on a quiet desktop:

```
Ran 3 steps.
Delta ratio: 100.0%
  step  1  delta     diff=0.000  trigger=none
  step  2  delta     diff=0.000  trigger=none
  step  3  delta     diff=0.000  trigger=none
```

3 of 3 steps classified DELTA, which means a paired VLM would receive small
crops instead of full frames — **75% token savings** per the default 1600 /
400 per-observation constants.

If that works, the rest of the library is composition.

## 4. Next steps by use case

- **"I want to drive a real app"** — swap `ScriptedModel` for
  [`ClaudeModel`](../deltavision_os/model/claude.py) or `OpenAIModel`. Set
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, point `--safety strict` before
  pointing at production machines, and start with small `--max-steps`.
- **"I want to benchmark on OSWorld"** — see [`benchmarks.md`](benchmarks.md).
  Requires Docker-KVM, ~10 GB VM image, about an hour of setup.
- **"Something broke"** — check [`troubleshooting.md`](troubleshooting.md).

## 5. What DeltaVision is doing under the hood

Reading the observation stream once manually builds intuition. Replace the
agent loop with one that just prints:

```python
async with platform:
    frame = await platform.capture()
    from deltavision_os.observation.builder import build_observation
    obs = build_observation(
        obs_type="full_frame", task="inspect", step=0, last_action=None,
        frame=frame, url="", trigger_reason="initial",
    )
    print(type(obs).__name__, obs.obs_type, obs.trigger_reason)
```

Then capture a second frame after a small change, compute the diff, and
hand it to the builder — you'll see it return a `DeltaObservation` instead.
The CV pipeline is in [`vision/diff.py`](../deltavision_os/vision/diff.py)
and [`vision/classifier.py`](../deltavision_os/vision/classifier.py). Everything
is plain Python functions with typed dataclasses — no magic.
