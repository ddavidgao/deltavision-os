"""
OSWorld benchmark runner for DeltaVision-OS.

Runs a subset of OSWorld tasks through our agent loop and records:
  - per-task success (via env.evaluate())
  - steps, delta_ratio, estimated token cost
  - DB row + artifact dir per run

Primary claim this runner backs up: "DeltaVision saves N% of tokens on OSWorld
with <M pp drop in task success." Run TWICE per model — once with default
config (DeltaVision gated) and once with FORCE_FULL_FRAME=True — and compare.

Usage:
    # Windows Ollama / llama.cpp endpoint via tunnel
    python benchmarks/run_osworld.py \\
        --oswo-repo ~/OSWorld \\
        --subset test_small.json \\
        --model ui-tars-q4km.gguf \\
        --adapter ui-tars \\
        --base-url http://127.0.0.1:8080/v1

    # Same, forced full-frame ablation
    python benchmarks/run_osworld.py ... --force-full-frame

Requires:
    - OSWorld cloned locally (https://github.com/xlang-ai/OSWorld)
    - `pip install -e .` run inside the OSWorld repo (its desktop_env package)
    - Docker (or VMware Workstation) installed and running
    - VM image downloaded (first env.reset() auto-pulls a ~30GB snapshot)

Known OSWorld pain points (document as you hit them):
    - macOS: VMware Fusion networking broken on Apple Silicon; run VM on Windows
    - First-run VM image pull is long; cache it once and reuse snapshots
    - Some test tasks score flaky; prefer test_small.json or test_nogdrive.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltavision_os.agent.loop import run_agent
from deltavision_os.capture.osworld import OSWorldPlatform
from deltavision_os.config import DeltaVisionConfig
from deltavision_os.safety import SafetyLayer


TOK_FULL_FRAME = 1600
TOK_DELTA = 400


# Categories known to be broken upstream — default skip list so `--max-tasks
# N` doesn't silently feed a naive user 0/N on the same known-bad path.
# See docs/troubleshooting.md for the bug links.
DEFAULT_SKIP_CATEGORIES = frozenset([
    "chrome",           # _chrome_open_tabs_setup uses sync Playwright in asyncio
    "libreoffice_calc", # AT-SPI tree generation hangs >10min (OSWorld#185)
])


def load_task_index(oswo_repo: Path, subset_name: str,
                    skip_categories: frozenset[str] = DEFAULT_SKIP_CATEGORIES,
                    categories: tuple[str, ...] = ()) -> list[dict]:
    """Resolve a subset file (e.g. 'test_small.json') into a list of task
    config dicts loaded from evaluation_examples/examples/{category}/{id}.json.

    Filters:
      skip_categories: categories known broken upstream (default: chrome +
          libreoffice_calc). Omit empty frozenset to disable and get raw order.
      categories: if non-empty, restrict to ONLY these categories. Useful
          for "one task per app" comprehensive-test scenarios — pass e.g.
          ('gimp', 'libreoffice_writer', 'vs_code') with --max-tasks 3.
    """
    root = oswo_repo / "evaluation_examples"
    subset_path = root / subset_name
    if not subset_path.exists():
        raise FileNotFoundError(
            f"OSWorld subset not found: {subset_path}. "
            f"Known subsets: test_small.json, test_nogdrive.json, test_all.json"
        )
    index = json.loads(subset_path.read_text())
    if not isinstance(index, dict):
        raise ValueError(f"Unexpected subset format in {subset_path}")

    examples = root / "examples"
    tasks = []
    missing = []
    skipped_count = 0
    for category, files in index.items():
        if category in skip_categories:
            skipped_count += len(files)
            continue
        if categories and category not in categories:
            continue
        for fname in files:
            p = examples / category / f"{fname}.json"
            if p.exists():
                tc = json.loads(p.read_text())
                tc.setdefault("id", fname)
                tc.setdefault("_category", category)
                tasks.append(tc)
            else:
                missing.append(str(p))
    if missing:
        print(f"WARN: {len(missing)} task files missing from "
              f"evaluation_examples/examples/ (first: {missing[0]})")
    if skipped_count:
        print(f"Skipped {skipped_count} tasks in {sorted(skip_categories)} "
              f"(known upstream bugs — override with --no-skip-default-broken).")
    return tasks


def build_model(args):
    # We only wire llamacpp / openai-compat here because OSWorld runs are
    # long and we want local inference. Extend later if needed.
    from deltavision_os.model.openai import OpenAIModel
    return OpenAIModel(
        model=args.model, base_url=args.base_url, api_key="not-needed"
    )


async def run_one(env, task_config, model, agent_config, safety) -> dict:
    """Run a single OSWorld task through our agent loop. Returns a per-task
    result dict. Does NOT raise on in-loop errors — captures them."""
    tid = task_config.get("id", "unknown")
    result = {
        "task_id": tid,
        "category": task_config.get("_category"),
        "instruction": task_config.get("instruction", ""),
        "steps": 0,
        "delta_ratio": 0.0,
        "new_page_count": 0,
        "score": None,
        "success": False,
        "error": None,
        "elapsed_s": 0.0,
        "transitions": [],
    }

    t0 = time.time()
    try:
        obs = env.reset(task_config=task_config)
        platform = OSWorldPlatform(env, initial_obs=obs)
        instruction = platform.instruction or task_config.get("instruction", "")
        result["instruction"] = instruction

        state = await run_agent(instruction, model, platform, agent_config,
                                safety=safety)
        result["steps"] = state.step
        result["delta_ratio"] = state.delta_ratio
        result["new_page_count"] = state.new_page_count
        result["transitions"] = state.transition_log

        score = platform.evaluate()
        result["score"] = float(score) if score is not None else 0.0
        result["success"] = bool(result["score"])
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def estimated_tokens(result: dict) -> int:
    if not result.get("transitions"):
        return 0
    t = TOK_FULL_FRAME  # initial full-frame observation
    for tr in result["transitions"]:
        t += TOK_DELTA if tr["transition"] == "delta" else TOK_FULL_FRAME
    return t


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oswo-repo", default=os.environ.get("OSWORLD_REPO", ""),
                    help="Path to an OSWorld git clone")
    ap.add_argument("--subset", default="test_small.json")
    ap.add_argument("--model", default="ui-tars-q4km.gguf")
    ap.add_argument("--adapter", default="ui-tars")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--provider", default="docker",
                    choices=("docker", "vmware"))
    ap.add_argument("--os-type", default="Ubuntu",
                    choices=("Ubuntu", "Windows"))
    ap.add_argument("--force-full-frame", action="store_true",
                    help="Ablation: disable DeltaVision gating")
    ap.add_argument("--a11y-hybrid", action="store_true",
                    help="Enable accessibility-tree hybrid observations "
                         "(require_a11y_tree=True on OSWorld env). Pairs "
                         "structured UI text with pixel crops; mitigates "
                         "UI-TARS-style early-termination on delta crops.")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="Cap at N tasks (0 = use whole subset)")
    ap.add_argument("--categories", default="",
                    help="Comma-separated list of task categories to "
                         "restrict to (e.g. gimp,libreoffice_writer,vs_code). "
                         "Empty = all categories.")
    ap.add_argument("--no-skip-default-broken", action="store_true",
                    help="Include chrome + libreoffice_calc tasks. Default "
                         "skips them because of known upstream bugs "
                         "(Playwright asyncio, AT-SPI Calc hang). See "
                         "docs/troubleshooting.md.")
    ap.add_argument("--no-safety", action="store_true",
                    help="Skip SafetyLayer (OSWorld is sandboxed)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not args.oswo_repo:
        raise SystemExit("--oswo-repo (or $OSWORLD_REPO) required")
    oswo_repo = Path(args.oswo_repo).expanduser().resolve()
    if not (oswo_repo / "desktop_env").exists():
        raise SystemExit(
            f"No desktop_env/ under {oswo_repo}. Did you clone OSWorld?"
        )

    skip = frozenset() if args.no_skip_default_broken else DEFAULT_SKIP_CATEGORIES
    cats = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    tasks = load_task_index(oswo_repo, args.subset,
                            skip_categories=skip, categories=cats)
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        raise SystemExit(f"No tasks to run (subset={args.subset}, "
                         f"skip={sorted(skip)}, cats={cats}). "
                         f"Try --no-skip-default-broken or pick other categories.")

    obs_mode = (
        "FORCED-FULL" if args.force_full_frame
        else ("A11Y-HYBRID" if args.a11y_hybrid else "PIXEL-DELTA")
    )
    print(f"Loaded {len(tasks)} tasks from {args.subset}")
    print(f"Model: {args.model}  base_url: {args.base_url}  "
          f"provider: {args.provider}  mode: {obs_mode}")

    # Import desktop_env from the cloned OSWorld repo
    sys.path.insert(0, str(oswo_repo))
    from desktop_env.desktop_env import DesktopEnv

    # OSWorld's docker provider resolves `./docker_vm_data/Ubuntu.qcow2.zip`
    # relative to cwd. If the user runs this script from a fresh clone, the
    # VM image is NOT reused and triggers an ~11GB re-download. Change cwd
    # to the OSWorld repo so the existing image is discovered.
    original_cwd = os.getcwd()
    os.chdir(str(oswo_repo))
    print(f"(chdir to {oswo_repo} so docker_vm_data/ is reused)")

    env = DesktopEnv(
        provider_name=args.provider,
        os_type=args.os_type,
        action_space="pyautogui",
        require_a11y_tree=args.a11y_hybrid,
    )
    model = build_model(args)
    agent_config = DeltaVisionConfig()
    if args.force_full_frame:
        agent_config.FORCE_FULL_FRAME = True
    safety = None if args.no_safety else SafetyLayer()

    results = []
    t_start = time.time()
    try:
        for i, tc in enumerate(tasks):
            print(f"\n--- Task {i+1}/{len(tasks)}: "
                  f"{tc.get('id', '?')}  [{tc.get('_category', '?')}] ---")
            r = await run_one(env, tc, model, agent_config, safety)
            r["estimated_tokens"] = estimated_tokens(r)
            results.append(r)
            print(f"  steps={r['steps']}  success={r['success']}  "
                  f"score={r['score']}  delta_ratio={r['delta_ratio']:.2f}  "
                  f"tokens={r['estimated_tokens']}  elapsed={r['elapsed_s']}s"
                  + (f"  ERR={r['error']}" if r["error"] else ""))
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"env.close() failed: {e}")

    # Aggregate
    n = len(results)
    succ = sum(1 for r in results if r["success"])
    total_tok = sum(r.get("estimated_tokens") or 0 for r in results)
    avg_steps = sum(r["steps"] for r in results) / max(n, 1)
    errors = sum(1 for r in results if r["error"])

    summary = {
        "subset": args.subset,
        "model": args.model,
        "adapter": args.adapter,
        "base_url": args.base_url,
        "provider": args.provider,
        "os_type": args.os_type,
        "force_full_frame": args.force_full_frame,
        "a11y_hybrid": args.a11y_hybrid,
        "obs_mode": obs_mode,
        "n": n,
        "success_count": succ,
        "success_rate": round(succ / max(n, 1), 4),
        "error_count": errors,
        "total_estimated_tokens": total_tok,
        "avg_steps": round(avg_steps, 2),
        "wall_time_s": round(time.time() - t_start, 1),
        "results": results,
    }

    print("\n=== OSWorld summary ===")
    print(f"  n={n}  success={succ}/{n} ({summary['success_rate']*100:.1f}%)"
          f"  errors={errors}  avg_steps={summary['avg_steps']}")
    print(f"  estimated_tokens_total={total_tok:,}"
          f"  wall={summary['wall_time_s']}s")

    mode = (
        "forced_full_frame" if args.force_full_frame
        else ("a11y_hybrid" if args.a11y_hybrid else "pixel_delta")
    )
    out = Path(args.output) if args.output else (
        Path(__file__).parent /
        f"osworld_{args.subset.replace('.json','')}_{mode}.json"
    )
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Artifact: {out}")

    # DB row + artifact dir
    from benchmarks._repro import save_run, snapshot_context
    config_snapshot = snapshot_context({
        "model": args.model,
        "adapter": args.adapter,
        "base_url": args.base_url,
        "subset": args.subset,
        "provider": args.provider,
        "os_type": args.os_type,
        "force_full_frame": args.force_full_frame,
        "a11y_hybrid": args.a11y_hybrid,
        "obs_mode": obs_mode,
        "max_tasks": args.max_tasks,
        "oswo_repo": str(oswo_repo),
    })
    metrics_for_db = {k: v for k, v in summary.items() if k != "results"}
    backend_slug = (
        f"{args.model}_force_full_frame" if args.force_full_frame
        else (f"{args.model}_a11y_hybrid" if args.a11y_hybrid
              else f"{args.model}_pixel_delta")
    )
    rid, run_dir = save_run(
        benchmark="osworld",
        backend=backend_slug,
        metrics=metrics_for_db,
        config=config_snapshot,
        notes=f"OSWorld {args.subset} n={n} success={succ}/{n} mode={obs_mode}",
        primary_artifact_path=out,
    )
    print(f"DB run id: {rid}    artifact dir: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
