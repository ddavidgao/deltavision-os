"""
V2 ablation: delta-gated vs forced-full-frame on a scripted desktop trajectory.

Mirrors V1's run_ablation.py but drives OSNativePlatform with a ScriptedModel
emitting a fixed action sequence. The actions (wait + Spotlight open/close)
are safe on any standard macOS setup and produce a realistic mix of DELTA and
NEW_PAGE transitions — enough to compare observation-token cost between the
two gating strategies.

What it measures: observation-only token cost. Not task-completion — the
"task" is a scripted trajectory, not a goal the model is trying to achieve.
This is the same kind of number as V1's paper claim (95% token reduction),
computed the same way (via the 1600/400 per-observation constants).

Usage:
    python benchmarks/run_ablation_os.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltavision_os.agent.loop import run_agent
from deltavision_os.agent.actions import Action, ActionType
from deltavision_os.capture.os_native import OSNativePlatform
from deltavision_os.config import DeltaVisionConfig
from deltavision_os.model.scripted import ScriptedModel


# Same constants V1 uses for its 95% claim, so the two numbers are
# methodologically comparable.
TOK_FULL_FRAME = 1600
TOK_DELTA = 400


def scripted_trajectory() -> list[Action]:
    """Safe Mac desktop trajectory: wait-cycles punctuated by Spotlight
    open/close. No file ops, no network, no app launches beyond Spotlight.
    """
    return [
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.HOTKEY, key="cmd+space"),
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.KEY, key="escape"),
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.HOTKEY, key="cmd+space"),
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.KEY, key="escape"),
        Action(type=ActionType.WAIT, duration_ms=500),
    ]


def natural_token_estimate(transition_log) -> dict:
    """Sum tokens based on what the classifier would have sent naturally.
    Initial full-frame observation is always counted (step 0)."""
    tokens = TOK_FULL_FRAME
    full = 1
    deltas = 0
    for entry in transition_log:
        if entry["transition"] == "delta":
            tokens += TOK_DELTA
            deltas += 1
        else:
            tokens += TOK_FULL_FRAME
            full += 1
    return {"total_tokens": tokens, "full_frames": full, "deltas": deltas}


def forced_full_token_estimate(step_count: int) -> dict:
    total = (step_count + 1) * TOK_FULL_FRAME
    return {"total_tokens": total, "full_frames": step_count + 1, "deltas": 0}


async def run_once(force_full: bool, trajectory: list[Action]) -> dict:
    config = DeltaVisionConfig()
    config.FORCE_FULL_FRAME = force_full
    platform = OSNativePlatform(cursor_park=None)
    model = ScriptedModel(trajectory)

    async with platform:
        t0 = time.time()
        state = await run_agent("ablation", model, platform, config)
        elapsed = time.time() - t0

    step_count = len(state.transition_log)
    natural = natural_token_estimate(state.transition_log)
    effective = forced_full_token_estimate(step_count) if force_full else natural

    return {
        "mode": "forced_full_frame" if force_full else "delta_gated",
        "elapsed_s": round(elapsed, 2),
        "steps": step_count,
        "transitions": state.transition_log,
        "natural_classification": natural,
        "effective_tokens": effective,
    }


async def main():
    trajectory = scripted_trajectory()
    print("DeltaVision-OS ablation: delta-gated vs forced full-frame\n")
    print(f"Trajectory: {len(trajectory)} scripted actions "
          "(WAIT + Spotlight open/close)\n")

    print("Run 1/2: delta-gated")
    delta_run = await run_once(False, trajectory)
    _print_run(delta_run)

    print("\nRun 2/2: forced full-frame")
    full_run = await run_once(True, trajectory)
    _print_run(full_run)

    delta_tok = delta_run["effective_tokens"]["total_tokens"]
    full_tok = full_run["effective_tokens"]["total_tokens"]
    saved = full_tok - delta_tok
    pct = (saved / full_tok * 100) if full_tok else 0.0

    print("\n--- A/B result ---")
    print(f"  Forced full-frame:     {full_tok:>6,} tokens "
          f"(over {full_run['steps'] + 1} observations)")
    print(f"  DeltaVision gated:     {delta_tok:>6,} tokens "
          f"({delta_run['effective_tokens']['full_frames']} full + "
          f"{delta_run['effective_tokens']['deltas']} delta)")
    print(f"  Saved:                 {saved:>6,} tokens ({pct:.1f}%)")
    print(f"  Delta ratio:           {delta_run['effective_tokens']['deltas']}/"
          f"{delta_run['steps']} steps "
          f"({delta_run['effective_tokens']['deltas'] / max(delta_run['steps'], 1) * 100:.0f}%)")

    out = Path(__file__).parent / "ablation_result.json"
    payload = {
        "delta_gated": delta_run,
        "forced_full_frame": full_run,
        "comparison": {
            "full_frame_tokens": full_tok,
            "delta_gated_tokens": delta_tok,
            "saved_tokens": saved,
            "saved_pct": round(pct, 1),
        },
        "constants": {
            "TOK_FULL_FRAME": TOK_FULL_FRAME,
            "TOK_DELTA": TOK_DELTA,
        },
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nArtifact: {out}")

    from benchmarks._repro import save_run, snapshot_context
    config = snapshot_context({
        "trajectory_actions": [str(a) for a in trajectory],
        "trajectory_length": len(trajectory),
        "platform": "os_native",
    })
    metrics_for_db = {
        "comparison": payload["comparison"],
        "constants": payload["constants"],
        "delta_steps": delta_run["steps"],
        "delta_ratio": (delta_run["effective_tokens"]["deltas"]
                       / max(delta_run["steps"], 1)),
        "token_cost": delta_tok,
    }
    rid, run_dir = save_run(
        benchmark="ablation_os",
        backend="scripted_spotlight",
        metrics=metrics_for_db,
        config=config,
        notes=f"Matched-trajectory A/B, saved={pct:.1f}%",
        primary_artifact_path=out,
        transition_log=delta_run["transitions"],
    )
    print(f"DB run id: {rid}    artifact dir: {run_dir}")


def _print_run(run: dict) -> None:
    eff = run["effective_tokens"]
    print(f"  {run['steps']} steps in {run['elapsed_s']}s  "
          f"full={eff['full_frames']} delta={eff['deltas']}  "
          f"tokens={eff['total_tokens']:,}")
    for t in run["transitions"]:
        print(f"    step {t['step']:2d}  {t['transition']:<8s}  "
              f"diff={t['diff_ratio']:.3f}  phash={t['phash_distance']:2d}  "
              f"trigger={t['trigger']}")


if __name__ == "__main__":
    asyncio.run(main())
