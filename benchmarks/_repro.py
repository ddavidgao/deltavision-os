"""
Shared reproducibility helpers for every benchmark in this directory.

Every benchmark run must:
  1. save to results/deltavision.db via ResultStore.save() → returns auto-increment run id
  2. create a per-run artifact directory: benchmarks/runs/{bench}_{backend}/run_{id}/
  3. snapshot the config (git SHA, python version, backend info, model name, parameters)
  4. verify the row exists before declaring done

This mirrors V1's reproducibility discipline (`~/Projects/deltavision/results/store.py`
and the benchmarks in `~/Projects/deltavision/benchmarks/ablation/`).
"""

import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def slug(s: str) -> str:
    """Safe directory slug from a model/backend name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower() or "unknown"


def snapshot_context(extra: Optional[dict] = None) -> dict:
    """Capture minimum reproducibility metadata.

    Returns dict with git_sha, git_dirty, python_version, platform, timestamp_iso.
    Merge `extra` on top for backend-specific info (ollama_version, llama_cpp_build, etc.).
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=REPO_ROOT
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, cwd=REPO_ROOT
        ).decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = None
    ctx = {
        "git_sha": sha,
        "git_dirty": dirty,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_iso": datetime.now().isoformat(),
    }
    if extra:
        ctx.update(extra)
    return ctx


def run_dir(benchmark: str, backend: str, run_id: int) -> Path:
    """Return the artifact directory for a run. Creates it."""
    d = RUNS_ROOT / f"{slug(benchmark)}_{slug(backend)}" / f"run_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(
    *,
    benchmark: str,
    backend: str,
    metrics: dict,
    config: dict,
    notes: str = "",
    primary_artifact_path: Optional[Path] = None,
    transition_log: Optional[list] = None,
) -> tuple[int, Path]:
    """Save to DB + create artifact dir + dump config snapshot. Returns (run_id, dir).

    primary_artifact_path: if given, copy it into the run dir as result.json.
    """
    from deltavision_os.results.store import ResultStore

    db = ResultStore()
    try:
        rid = db.save(
            benchmark=benchmark,
            backend=backend,
            metrics=metrics,
            config=config,
            transition_log=transition_log or [],
            notes=notes,
        )
        verify = db.query("SELECT id FROM runs WHERE id = ?", (rid,))
        if not verify:
            raise RuntimeError(f"DB save verification failed for run {rid}")
    finally:
        db.close()

    d = run_dir(benchmark, backend, rid)
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    if primary_artifact_path is not None and primary_artifact_path.exists():
        shutil.copy2(primary_artifact_path, d / "result.json")

    return rid, d


def backfill_run(
    *,
    benchmark: str,
    backend: str,
    result_json_path: Path,
    notes: str = "backfilled",
) -> Optional[int]:
    """Register a historical result JSON as a DB row.

    The JSON is parsed, flattened to a metrics dict, and saved. The file is also
    copied into the new artifact dir for cross-reference.
    """
    if not result_json_path.exists():
        print(f"  SKIP (missing): {result_json_path}")
        return None
    data = json.loads(result_json_path.read_text())
    config = {
        "backfill_source": str(result_json_path),
        "backfill_timestamp": datetime.now().isoformat(),
    }
    # Try to pull config out of the result JSON if present
    for key in ("model", "adapter", "base_url", "per_platform_limit", "mode",
                "thresholds", "constants"):
        if key in data:
            config[key] = data[key]
    rid, d = save_run(
        benchmark=benchmark,
        backend=backend,
        metrics=data,
        config=config,
        notes=notes,
        primary_artifact_path=result_json_path,
    )
    return rid
