"""
Desktop idle observation benchmark.

Captures the Mac (or Linux/Windows) desktop N times at 1Hz, runs the full
CV pipeline on each pair, and reports the classifier's decisions + delta
ratio.

What this proves:
  - V2's OSNativePlatform captures real frames reliably
  - The V1-derived CV pipeline works without any browser or URL
  - On a quiet desktop, the classifier correctly identifies each step as
    a DELTA (no false NEW_PAGE triggers)
  - Token savings are achievable even on desktops — if we had a model
    attached, 9/10 steps would send ~400 tokens instead of ~1600

Usage:
    python benchmarks/desktop_idle_observe.py --rounds 10 --interval 1.0

No model API calls — this is pure CV benchmarking, free to run.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Allow running as `python benchmarks/desktop_idle_observe.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltavision_os.vision.diff import compute_diff
from deltavision_os.vision.classifier import classify_transition, extract_anchor, TransitionType
from deltavision_os.capture.os_native import OSNativePlatform
from deltavision_os.config import DeltaVisionConfig


async def run(rounds: int, interval: float, save_dir: Path | None):
    config = DeltaVisionConfig()
    platform = OSNativePlatform(cursor_park=None)

    async with platform:
        # Anchor frame
        t0 = await platform.capture()
        anchor = extract_anchor(t0, config)
        url_t0 = None

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            t0.save(save_dir / "step_000.png")

        transitions = []
        start = time.time()

        for i in range(1, rounds + 1):
            await asyncio.sleep(interval)

            t1 = await platform.capture()
            diff = compute_diff(t0, t1, config)
            cls = classify_transition(
                t0=t0,
                t1=t1,
                url_before="",
                url_after="",
                anchor_template=anchor,
                config=config,
                diff_result=diff,
                last_action_type="wait",
            )

            transitions.append({
                "step": i,
                "transition": cls.transition.value,
                "trigger": cls.trigger,
                "diff_ratio": cls.diff_ratio,
                "phash_distance": cls.phash_distance,
                "anchor_score": cls.anchor_score,
            })

            print(
                f"step {i:3d}  {cls.transition.value:<8s}  "
                f"diff={cls.diff_ratio:5.3f}  "
                f"phash={cls.phash_distance:2d}  "
                f"anchor={cls.anchor_score:.2f}  "
                f"trigger={cls.trigger}"
            )

            if save_dir:
                t1.save(save_dir / f"step_{i:03d}.png")

            # On NEW_PAGE, re-anchor (same invariant as the agent loop)
            if cls.transition == TransitionType.NEW_PAGE:
                t0 = t1
                anchor = extract_anchor(t0, config)

        elapsed = time.time() - start
        deltas = sum(1 for t in transitions if t["transition"] == "delta")
        ratio = deltas / len(transitions) if transitions else 0.0

        print()
        print(f"Observed {rounds} steps in {elapsed:.1f}s")
        print(f"DELTA:   {deltas:3d} ({ratio:.1%})")
        print(f"NEW_PAGE:{rounds - deltas:3d}")
        print()
        print(f"Token savings if paired with a VLM at 1600 tok/full_frame, ~400 tok/delta:")
        saved = deltas * (1600 - 400)
        full_cost = rounds * 1600
        print(f"  Full frame every step: {full_cost:>6,} tokens")
        print(f"  DeltaVision gated:     {full_cost - saved:>6,} tokens ({saved:,} saved)")

        return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Optional: save each captured frame here")
    args = parser.parse_args()

    asyncio.run(run(args.rounds, args.interval, args.save_dir))


if __name__ == "__main__":
    main()
