"""
Persistent HTTP wrapper around an OSWorld DesktopEnv, so an external
agent (Claude Code subagent, in our case) can drive a task one step at a
time without rebuilding the env per step.

Endpoints (all return JSON):

    POST /init    body: {"task_config": {...}, "mode": "full" | "delta"}
                  → {"image_paths": [...], "instruction": "...", "step": 0}

    POST /step    body: {"action": "pyautogui.click(...)"}
                  → {"image_paths": [...], "step": N, "done": bool, "delta_ratio": float}

    POST /score   → {"score": float}

    POST /close   → {"closed": true}

`image_paths` are absolute Windows-side paths under
`C:/Users/david/deltavision-os/_osworld_run/` so the subagent can SCP them
back to Mac and Read them with vision.

For mode="delta", we wrap the raw screenshot through our existing CV pipeline
to produce diff heatmap + crop pairs, matching what the agent loop normally
sends to the model.

Run with:
    python3 benchmarks/osworld_cli_server.py --port 5500
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop_env.desktop_env import DesktopEnv  # provided by OSWorld in WSL venv

from deltavision_os.config import DeltaVisionConfig
from deltavision_os.vision.diff import compute_diff, extract_crops
from deltavision_os.vision.classifier import classify_transition, extract_anchor, TransitionType


OUT_DIR = Path("/mnt/c/Users/david/deltavision-os/_osworld_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WIN_OUT = "C:/Users/david/deltavision-os/_osworld_run"  # Windows-style path


class Session:
    """Holds the per-task state. Single-tenant — only one task at a time."""

    def __init__(self):
        self.env: Optional[DesktopEnv] = None
        self.mode: str = "full"
        self.config = DeltaVisionConfig()
        self.step_n: int = 0
        self.t0: Optional[Image.Image] = None  # baseline frame for delta
        self.anchor = None
        self.delta_decisions: list[str] = []
        self.last_action_str: str = ""

    def reset(self, task_config: dict, mode: str):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.mode = mode
        self.config = DeltaVisionConfig()
        self.env = DesktopEnv(
            provider_name="docker",
            os_type="Ubuntu",
            action_space="pyautogui",
            require_a11y_tree=False,
        )
        obs = self.env.reset(task_config=task_config)
        self.step_n = 0
        self.t0 = _pil(obs["screenshot"])
        self.anchor = extract_anchor(self.t0, self.config)
        self.delta_decisions = []
        self.last_action_str = ""

        full_path = OUT_DIR / "step_000_full.png"
        self.t0.save(full_path)
        return {
            "image_paths": [_winpath(full_path)],
            "instruction": obs.get("instruction") or "",
            "step": 0,
            "obs_kind": "full_frame",
            "trigger": "initial",
        }

    def step(self, action_str: str):
        if self.env is None:
            raise RuntimeError("call /init first")
        self.last_action_str = action_str
        obs, _reward, done, _info = self.env.step(action_str, pause=2)
        self.step_n += 1
        t1 = _pil(obs["screenshot"])

        # CV classification (always done for telemetry; mode dictates output).
        diff_result = compute_diff(self.t0, t1, self.config)
        cls = classify_transition(
            t0=self.t0, t1=t1,
            url_before="", url_after="",
            anchor_template=self.anchor, config=self.config,
            diff_result=diff_result, last_action_type="click",  # generic
        )
        self.delta_decisions.append({
            "step": self.step_n,
            "transition": cls.transition.value,
            "trigger": cls.trigger,
            "diff_ratio": cls.diff_ratio,
            "phash": cls.phash_distance,
        })

        out_paths = []
        force_full = self.mode == "full"
        if force_full or cls.transition == TransitionType.NEW_PAGE:
            full_path = OUT_DIR / f"step_{self.step_n:03d}_full.png"
            t1.save(full_path)
            out_paths.append(_winpath(full_path))
            obs_kind = "full_frame"
            self.t0 = t1
            self.anchor = extract_anchor(self.t0, self.config)
        else:
            # DELTA observation: save diff heatmap + crop pairs
            diff_path = OUT_DIR / f"step_{self.step_n:03d}_diff.png"
            diff_result.diff_image.save(diff_path)
            out_paths.append(_winpath(diff_path))
            crops = extract_crops(self.t0, t1,
                                  diff_result.changed_bboxes,
                                  self.config.CROP_PADDING)
            for i, c in enumerate(crops):
                bp = OUT_DIR / f"step_{self.step_n:03d}_crop{i}_before.png"
                ap = OUT_DIR / f"step_{self.step_n:03d}_crop{i}_after.png"
                c["crop_before"].save(bp)
                c["crop_after"].save(ap)
                out_paths.append(_winpath(bp))
                out_paths.append(_winpath(ap))
            obs_kind = "delta"

        deltas = sum(1 for d in self.delta_decisions if d["transition"] == "delta")
        delta_ratio = deltas / max(self.step_n, 1)

        return {
            "image_paths": out_paths,
            "step": self.step_n,
            "done": bool(done),
            "obs_kind": obs_kind,
            "trigger": cls.trigger,
            "diff_ratio": round(cls.diff_ratio, 3),
            "delta_ratio_so_far": round(delta_ratio, 3),
            "natural_classification": cls.transition.value,
        }

    def evaluate(self) -> float:
        if self.env is None:
            raise RuntimeError("call /init first")
        return float(self.env.evaluate())

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


SESSION = Session()
app = Flask(__name__)


def _pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _winpath(p: Path) -> str:
    """Convert /mnt/c/... to C:/... (Windows path the subagent SCPs from)."""
    s = str(p)
    if s.startswith("/mnt/c/"):
        return "C:/" + s[len("/mnt/c/"):]
    return s


@app.post("/init")
def init():
    body = request.get_json()
    task_config = body["task_config"]
    mode = body.get("mode", "full")
    if mode not in ("full", "delta"):
        return jsonify({"error": f"mode must be full or delta, got {mode}"}), 400
    return jsonify(SESSION.reset(task_config, mode))


@app.post("/step")
def step():
    body = request.get_json()
    return jsonify(SESSION.step(body["action"]))


@app.post("/patch_evaluator")
def patch_evaluator():
    """Patch the evaluator config in the running env (for when /init was called with incomplete config).
    Also rebuilds result_getter / expected_getter so evaluate() works correctly.
    """
    import traceback as _tb
    if SESSION.env is None:
        return jsonify({"error": "no active session"}), 400
    try:
        patch = request.get_json()
        SESSION.env.evaluator.update(patch)
        env = SESSION.env
        evaluator = env.evaluator
        # Re-run the getter binding from _set_evaluator_info logic
        from desktop_env.evaluators import getters as _getters
        if "result" in evaluator and evaluator["result"]:
            res = evaluator["result"]
            if isinstance(res, list):
                env.result_getter = [getattr(_getters, "get_{:}".format(r["type"])) for r in res]
            else:
                env.result_getter = getattr(_getters, "get_{:}".format(res["type"]))
        if "expected" in evaluator and evaluator["expected"]:
            exp = evaluator["expected"]
            if isinstance(exp, list):
                env.expected_getter = [getattr(_getters, "get_{:}".format(e["type"])) for e in exp]
            else:
                env.expected_getter = getattr(_getters, "get_{:}".format(exp["type"]))
        return jsonify({"patched": evaluator,
                        "result_getter": str(env.result_getter),
                        "expected_getter": str(getattr(env, "expected_getter", None))})
    except Exception as _e:
        return jsonify({"error": str(_e), "traceback": _tb.format_exc()}), 500


@app.post("/score")
def score():
    import traceback as _tb
    try:
        return jsonify({"score": SESSION.evaluate()})
    except Exception as _e:
        err = _tb.format_exc()
        return jsonify({"error": str(_e), "traceback": err}), 500


@app.post("/close")
def close():
    SESSION.close()
    return jsonify({"closed": True})


@app.get("/health")
def health():
    return jsonify({"ok": True, "step": SESSION.step_n,
                    "active": SESSION.env is not None,
                    "mode": SESSION.mode})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5500)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
