"""
One-time: register pre-existing JSON result artifacts as DB rows so every
historical run is queryable via SQLite.

For each known JSON in benchmarks/, create a runs-table row with
benchmark + backend + metrics_json + a stub config (marking it as backfilled).
The original JSON is also copied into benchmarks/runs/{bench}_{backend}/run_{id}/
for cross-reference.

Safe to run multiple times: each invocation creates NEW rows. If you want to
dedupe, query the DB and delete backfill rows before re-running.

Usage:
    python benchmarks/backfill_runs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._repro import backfill_run


BF = Path(__file__).resolve().parent

REGISTRATIONS = [
    # (benchmark, backend, json path, notes)
    ("ablation_os", "scripted_spotlight",
     BF / "ablation_result.json",
     "Initial matched-trajectory A/B, 68.2% savings"),
    ("ablation_sweep", "scripted_3traj_3thresh",
     BF / "ablation_sweep_result.json",
     "3 trajectories x 3 thresholds sweep, pre-retrofit"),
    ("screenspot_v2", "qwen2_5vl_7b_q4_ollama",
     BF / "screenspot_qwen_n90.json",
     "Qwen2.5-VL-7B Q4 via Ollama, n=90"),
    ("screenspot_v2", "qwen2_5vl_7b_q4_ollama_n15",
     BF / "screenspot_result_qwen2_5vl_7b.json",
     "Qwen2.5-VL-7B Q4 smoke test, n=15"),
    ("screenspot_v2", "claude_sonnet_4_6_subagent",
     BF / "screenspot_sonnet_n30.json",
     "Claude Sonnet 4.6 via subagent (clean, no ground-truth leak), n=30"),
    ("screenspot_v2", "uitars_1_5_7b_q4_llamacpp",
     BF / "screenspot_uitars_llamacpp_n90.json",
     "UI-TARS-1.5-7B Q4 via llama.cpp, n=90"),
    ("screenspot_v2", "uitars_1_5_7b_q4_llamacpp_n30",
     BF / "screenspot_uitars_llamacpp_n30.json",
     "UI-TARS-1.5-7B Q4 smoke, n=30"),
]


def main():
    created = []
    for bench, backend, path, notes in REGISTRATIONS:
        rid = backfill_run(benchmark=bench, backend=backend,
                           result_json_path=path, notes=notes)
        if rid is not None:
            created.append((rid, bench, backend, str(path.name)))

    print(f"\nBackfilled {len(created)} rows:")
    for rid, bench, backend, fname in created:
        print(f"  #{rid:3d}  {bench:<20s}  {backend:<40s}  {fname}")


if __name__ == "__main__":
    main()
