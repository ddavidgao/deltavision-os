"""
CV pipeline performance benchmark.

Measures the end-to-end cost of the DeltaVision CV layer on real desktop
captures: diff computation + 4-layer classification + crop extraction.
This is the overhead added before any model inference runs.

What we're measuring:
  - capture   : mss screenshot latency
  - diff      : compute_diff (numpy threshold + morphological ops + contours)
  - classify  : 4-layer cascade (URL / diff ratio / pHash / anchor)
  - crops     : extract_crops from diff bboxes
  - total     : sum of above (= "CV pipeline overhead per step")

This is what needs to stay small (~40ms target) so the pipeline doesn't
become the bottleneck vs model inference (~1-10s per step).

Usage:
    python benchmarks/pipeline_perf.py --iterations 20
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltavision_os.vision.diff import compute_diff, extract_crops
from deltavision_os.vision.classifier import classify_transition, extract_anchor
from deltavision_os.capture.os_native import OSNativePlatform
from deltavision_os.config import DeltaVisionConfig


async def run(iterations: int):
    config = DeltaVisionConfig()
    platform = OSNativePlatform(cursor_park=None)

    async with platform:
        # Warm up — first capture is often slower (driver init)
        await platform.capture()
        t0 = await platform.capture()
        anchor = extract_anchor(t0, config)

        capture_ms = []
        diff_ms = []
        classify_ms = []
        crop_ms = []
        total_ms = []

        for i in range(iterations):
            s0 = time.perf_counter()
            t1 = await platform.capture()
            s1 = time.perf_counter()

            diff = compute_diff(t0, t1, config)
            s2 = time.perf_counter()

            cls = classify_transition(
                t0=t0, t1=t1,
                url_before="", url_after="",
                anchor_template=anchor, config=config,
                diff_result=diff, last_action_type="wait",
            )
            s3 = time.perf_counter()

            _ = extract_crops(t0, t1, diff.changed_bboxes, config.CROP_PADDING)
            s4 = time.perf_counter()

            capture_ms.append((s1 - s0) * 1000)
            diff_ms.append((s2 - s1) * 1000)
            classify_ms.append((s3 - s2) * 1000)
            crop_ms.append((s4 - s3) * 1000)
            total_ms.append((s4 - s0) * 1000)

        print(f"Captured {iterations} rounds.  Screen: {t0.width}x{t0.height}")
        print()
        print(f"{'stage':<12} {'min':>8} {'med':>8} {'p95':>8} {'max':>8}  (ms)")
        print("-" * 52)
        for name, vals in [
            ("capture",  capture_ms),
            ("diff",     diff_ms),
            ("classify", classify_ms),
            ("crop",     crop_ms),
            ("TOTAL",    total_ms),
        ]:
            vals_s = sorted(vals)
            p95 = vals_s[int(len(vals_s) * 0.95)]
            print(f"{name:<12} {min(vals):>7.1f} {statistics.median(vals):>7.1f} "
                  f"{p95:>7.1f} {max(vals):>7.1f}")

        print()
        print("Interpretation:")
        print(f"  Model inference is typically 1-10 seconds per step.")
        print(f"  CV pipeline overhead (TOTAL) should stay under ~100ms.")
        print(f"  At {statistics.median(total_ms):.0f}ms median, CV adds "
              f"{statistics.median(total_ms) / 1000:.3%} of a 1s inference window.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    asyncio.run(run(args.iterations))


if __name__ == "__main__":
    main()
