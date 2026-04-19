"""
V2 ablation sweep: robustness of the token-savings claim across trajectories
and classifier thresholds.

Extends run_ablation_os.py along two axes:
  1. Three scripted trajectories with different transition intensities:
       - idle_only:             10 waits, minimal screen change
       - spotlight_cycle:       Spotlight open/close x2 (moderate transitions)
       - mission_control_cycle: Mission Control open/close x2 (large transitions)
  2. Three NEW_PAGE_DIFF_THRESHOLD values: 0.30, 0.50, 0.75 (default).

For each (trajectory, threshold) pair we measure delta-gated token cost and
compare to a single forced-full-frame reference run for the same trajectory.
Produces a 3x3 savings-% matrix. Flat across the matrix → robust claim.
Wild variation → threshold-sensitive, flag in paper.

Usage:
    python benchmarks/run_ablation_sweep.py
Takes ~3 minutes (12 runs x ~15s each on macOS + Spotlight/Mission Control).
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


TOK_FULL_FRAME = 1600
TOK_DELTA = 400

DIFF_THRESHOLDS = [0.30, 0.50, 0.75]


def traj_idle_only() -> list[Action]:
    return [Action(type=ActionType.WAIT, duration_ms=500) for _ in range(10)]


def traj_spotlight() -> list[Action]:
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


def traj_mission_control() -> list[Action]:
    return [
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.WAIT, duration_ms=500),
        Action(type=ActionType.HOTKEY, key="ctrl+up"),
        Action(type=ActionType.WAIT, duration_ms=800),
        Action(type=ActionType.KEY, key="escape"),
        Action(type=ActionType.WAIT, duration_ms=800),
        Action(type=ActionType.HOTKEY, key="ctrl+up"),
        Action(type=ActionType.WAIT, duration_ms=800),
        Action(type=ActionType.KEY, key="escape"),
        Action(type=ActionType.WAIT, duration_ms=500),
    ]


TRAJECTORIES = {
    "idle_only": traj_idle_only,
    "spotlight_cycle": traj_spotlight,
    "mission_control_cycle": traj_mission_control,
}


async def run_one(trajectory: list[Action], force_full: bool,
                  diff_threshold: float) -> list[dict]:
    config = DeltaVisionConfig()
    config.FORCE_FULL_FRAME = force_full
    config.NEW_PAGE_DIFF_THRESHOLD = diff_threshold
    platform = OSNativePlatform(cursor_park=None)
    model = ScriptedModel(trajectory)
    async with platform:
        state = await run_agent("ablation", model, platform, config)
    return state.transition_log


def tokens_for(log: list[dict], force_full: bool) -> int:
    if force_full:
        return (len(log) + 1) * TOK_FULL_FRAME
    total = TOK_FULL_FRAME  # initial full-frame observation
    for entry in log:
        total += TOK_DELTA if entry["transition"] == "delta" else TOK_FULL_FRAME
    return total


async def main():
    t_start = time.time()
    results = {}

    for tname, traj_fn in TRAJECTORIES.items():
        print(f"\n=== trajectory: {tname} ({len(traj_fn())} steps) ===")

        # One forced-full-frame reference run per trajectory
        full_log = await run_one(traj_fn(), force_full=True,
                                 diff_threshold=DIFF_THRESHOLDS[-1])
        full_cost = tokens_for(full_log, force_full=True)
        print(f"  forced-full-frame reference: {full_cost:,} tokens "
              f"({len(full_log) + 1} observations)")

        traj_results = {"full_frame_reference_tokens": full_cost,
                        "sweep": {}}

        for th in DIFF_THRESHOLDS:
            delta_log = await run_one(traj_fn(), force_full=False,
                                      diff_threshold=th)
            delta_cost = tokens_for(delta_log, force_full=False)
            new_pages = sum(1 for e in delta_log if e["transition"] == "new_page")
            saved_pct = (full_cost - delta_cost) / full_cost * 100
            print(f"  threshold={th:.2f}:  {delta_cost:>6,} tokens  "
                  f"new_page={new_pages:2d}  savings={saved_pct:5.1f}%")

            traj_results["sweep"][f"{th:.2f}"] = {
                "delta_tokens": delta_cost,
                "new_pages": new_pages,
                "savings_pct": round(saved_pct, 1),
                "transitions": delta_log,
            }

        results[tname] = traj_results

    # Summary matrix
    print("\n=== Summary: token savings (%) by trajectory x NEW_PAGE_DIFF_THRESHOLD ===\n")
    header = "  " + f"{'trajectory':<25s}" + "".join(
        f"{'th=' + f'{th:.2f}':>10s}" for th in DIFF_THRESHOLDS
    )
    print(header)
    print("  " + "-" * (25 + 10 * len(DIFF_THRESHOLDS)))
    for tname in TRAJECTORIES:
        cells = [
            f"{results[tname]['sweep'][f'{th:.2f}']['savings_pct']:>9.1f}%"
            for th in DIFF_THRESHOLDS
        ]
        print("  " + f"{tname:<25s}" + "".join(cells))

    elapsed = time.time() - t_start
    print(f"\nTotal wall time: {elapsed:.1f}s")

    out = Path(__file__).parent / "ablation_sweep_result.json"
    payload = {
        "thresholds": DIFF_THRESHOLDS,
        "trajectories": results,
        "constants": {
            "TOK_FULL_FRAME": TOK_FULL_FRAME,
            "TOK_DELTA": TOK_DELTA,
        },
        "wall_time_s": round(elapsed, 1),
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Artifact: {out}")

    from benchmarks._repro import save_run, snapshot_context
    config = snapshot_context({
        "thresholds_swept": DIFF_THRESHOLDS,
        "trajectories": list(TRAJECTORIES.keys()),
        "platform": "os_native",
    })
    savings_matrix = {
        tname: {th_str: r["savings_pct"]
                for th_str, r in tdata["sweep"].items()}
        for tname, tdata in results.items()
    }
    metrics_for_db = {
        "savings_matrix": savings_matrix,
        "constants": payload["constants"],
        "wall_time_s": payload["wall_time_s"],
    }
    rid, run_dir = save_run(
        benchmark="ablation_sweep",
        backend="scripted_3traj_3thresh",
        metrics=metrics_for_db,
        config=config,
        notes=("3 trajectories x 3 NEW_PAGE_DIFF_THRESHOLDS, "
               "savings_matrix for robustness analysis"),
        primary_artifact_path=out,
    )
    print(f"DB run id: {rid}    artifact dir: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
