"""
Classifier sensitivity benchmark.

Sweeps synthetic "damage" levels from 0% to 100% pixel change and reports
how the 4-layer cascade responds. This is the CV side of DeltaVision's
story: what does the classifier actually decide given a known diff?

Useful for:
  - Tuning thresholds for your specific workload
  - Proving the classifier's decision boundary is where you think it is
  - Generating paper figures (cascade response curve)

No model needed — pure CV.

Usage:
    python benchmarks/classifier_sensitivity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from deltavision_os.config import DeltaVisionConfig
from deltavision_os.vision.diff import compute_diff
from deltavision_os.vision.classifier import classify_transition, extract_anchor


def make_base_frame(size=(800, 600)) -> Image.Image:
    """Noise-textured base frame. More realistic than a solid color —
    has real pHash structure and non-trivial compression characteristics."""
    rng = np.random.default_rng(42)
    arr = rng.integers(50, 200, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def damage(base: Image.Image, fraction: float, seed: int = 0) -> Image.Image:
    """Replace `fraction` of pixels (by area) in `base` with noise.

    Uses a contiguous block (like a real UI element changing) rather than
    random-sprinkle damage, so the diff engine's morphological pass behaves
    as it would on a real screenshot.
    """
    arr = np.asarray(base).copy()
    h, w = arr.shape[:2]

    # Calculate block dimensions for the target area fraction
    target_pixels = int(h * w * fraction)
    # Make the damaged region roughly square
    block_side = int(np.sqrt(target_pixels))
    block_side = min(block_side, min(h, w) - 4)
    block_side = max(block_side, 4)

    # Place the block randomly but deterministically per seed
    rng = np.random.default_rng(seed)
    y0 = rng.integers(0, max(1, h - block_side))
    x0 = rng.integers(0, max(1, w - block_side))

    replacement = rng.integers(0, 255, size=(block_side, block_side, 3), dtype=np.uint8)
    arr[y0:y0 + block_side, x0:x0 + block_side] = replacement
    return Image.fromarray(arr)


def run():
    config = DeltaVisionConfig()
    base = make_base_frame()
    anchor = extract_anchor(base, config)

    levels = [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 0.90, 0.99]

    print(
        f"{'damage':>8} {'diff_ratio':>10} {'phash':>6} {'anchor':>7} "
        f"{'transition':>10} {'trigger':>12}"
    )
    print("-" * 70)

    for f in levels:
        t1 = damage(base, f)
        diff = compute_diff(base, t1, config)
        cls = classify_transition(
            t0=base, t1=t1,
            url_before="", url_after="",
            anchor_template=anchor, config=config,
            diff_result=diff, last_action_type="click",
        )
        print(
            f"{f * 100:7.1f}% {cls.diff_ratio:10.3f} {cls.phash_distance:>6d} "
            f"{cls.anchor_score:>7.2f} {cls.transition.value:>10s} {cls.trigger:>12s}"
        )

    print()
    print("Interpretation:")
    print(f"  NEW_PAGE threshold (diff_ratio): {config.NEW_PAGE_DIFF_THRESHOLD}")
    print(f"  pHash distance threshold:         {config.PHASH_DISTANCE_THRESHOLD}")
    print(f"  Anchor match threshold:           {config.ANCHOR_MATCH_THRESHOLD}")
    print()
    print("Expected pattern: small damage stays DELTA; above ~75% diff the")
    print("classifier flips to NEW_PAGE via Layer 2 (diff_ratio).")


if __name__ == "__main__":
    run()
