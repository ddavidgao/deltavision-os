"""
V2 vision package: pure CV pipeline, model-agnostic and platform-agnostic.

Note on V1 divergence: V1 had `vision.capture` with Playwright-specific
screenshot helpers. In V2, capture is a Platform-level concern — see
`capture/` at the project root.
"""

from .diff import compute_diff, extract_crops, DiffResult
from .classifier import classify_transition, TransitionType, ClassificationResult
from .phash import compute_phash, hamming_distance
