"""
Record a live V2 demo run and save all intermediate artifacts for video.

For each step, we save:
  - Raw Mac capture (PNG)
  - DV thumbnail with green bboxes (PNG)
  - Crop(s) sent to the model (PNG)
  - Full diff heatmap (PNG)
  - Classifier decision (JSON)
  - Model response (JSON)

Output: benchmarks/demo_run/step_NN/*.{png,json}

Usage:
    python benchmarks/record_live_demo.py --steps 5 --model qwen2.5vl:7b \\
        --host 127.0.0.1 --port 11434

The next script (`render_demo_video.py`) reads this directory and builds
the visual demo video.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from deltavision_os.vision.diff import compute_diff, extract_crops
from deltavision_os.vision.classifier import classify_transition, extract_anchor
from deltavision_os.capture.os_native import OSNativePlatform
from deltavision_os.config import DeltaVisionConfig
from deltavision_os.model.ollama import OllamaModel
from deltavision_os.observation.builder import build_observation


async def run(n_steps: int, out_dir: Path, host: str, port: int, model_name: str, task: str):
    config = DeltaVisionConfig(MAX_STEPS=n_steps, POST_ACTION_WAIT_MS=300)
    platform = OSNativePlatform(cursor_park=None)

    # Real VLM — same thing the agent loop uses
    model = OllamaModel(model=model_name, host=f"http://{host}:{port}")

    out_dir.mkdir(parents=True, exist_ok=True)

    async with platform:
        t0 = await platform.capture()
        anchor = extract_anchor(t0, config)

        # Step 0: initial capture + model call
        step0 = out_dir / "step_00"
        step0.mkdir(exist_ok=True)
        t0.save(step0 / "capture.png")

        obs0 = build_observation(
            obs_type="full_frame",
            task=task, step=0, last_action=None,
            frame=t0, url="",
            trigger_reason="initial",
        )

        from deltavision_os.agent.state import AgentState
        state = AgentState(task=task)
        state.add_observation(obs0)

        print(f"[step 0] initial capture {t0.width}x{t0.height}, calling model...")
        t_model_start = time.time()
        resp0 = await model.predict(obs0, state)
        t_model_s = time.time() - t_model_start

        (step0 / "model.json").write_text(json.dumps({
            "obs_type": "full_frame",
            "trigger_reason": "initial",
            "model_time_s": t_model_s,
            "action": str(resp0.action) if resp0.action else None,
            "done": resp0.done,
            "reasoning": resp0.reasoning[:500],
            "confidence": resp0.confidence,
        }, indent=2))
        (step0 / "classify.json").write_text(json.dumps({
            "step": 0,
            "obs_type": "full_frame",
            "trigger": "initial",
            "diff_ratio": 0.0,
            "phash_distance": 0,
            "anchor_score": 1.0,
            "estimated_tokens": 1600,  # full frame baseline
        }, indent=2))

        if resp0.done or resp0.action is None:
            print(f"  model finished at step 0: {resp0.reasoning[:120]}")
            return

        # Execute the first action — if it's a WAIT, nothing to actually do
        state.add_response(resp0)

        for i in range(1, n_steps + 1):
            step_dir = out_dir / f"step_{i:02d}"
            step_dir.mkdir(exist_ok=True)

            # simulate the action's effect by just waiting (we don't actually
            # drive the cursor in this demo — safe for a recording session)
            await asyncio.sleep(config.POST_ACTION_WAIT_MS / 1000)

            t1 = await platform.capture()
            diff = compute_diff(t0, t1, config)
            cls = classify_transition(
                t0=t0, t1=t1,
                url_before="", url_after="",
                anchor_template=anchor, config=config,
                diff_result=diff, last_action_type="wait",
            )

            # Build the thumbnail+crops the model will see on a DELTA step
            crops = extract_crops(t0, t1, diff.changed_bboxes, config.CROP_PADDING)

            # Thumbnail with green boxes
            thumb = t1.resize((320, 225), Image.LANCZOS)
            draw = ImageDraw.Draw(thumb)
            sx = 320 / t1.width
            sy = 225 / t1.height
            for c in crops:
                x, y, w, h = c["bbox"]
                draw.rectangle([
                    (int(x * sx) - 1, int(y * sy) - 1),
                    (int((x + w) * sx) + 1, int((y + h) * sy) + 1),
                ], outline=(0, 255, 0), width=2)
            thumb.save(step_dir / "thumb.png")

            t1.save(step_dir / "capture.png")
            if diff.diff_image is not None:
                diff.diff_image.save(step_dir / "diff.png")

            # Crops (up to first 2)
            for ci, c in enumerate(crops[:2]):
                c["crop_after"].save(step_dir / f"crop_{ci}_after.png")

            # Build the observation that would be sent
            from deltavision_os.vision.classifier import TransitionType as _TT
            is_new_page = cls.transition == _TT.NEW_PAGE
            obs = build_observation(
                obs_type="full_frame" if is_new_page else "delta",
                task=task, step=i,
                last_action=resp0.action,  # from previous step
                frame=t1 if is_new_page else None,
                diff_result=diff if not is_new_page else None,
                crops=crops if not is_new_page else None,
                action_had_effect=diff.action_had_effect,
                no_change_count=0,
                trigger_reason=cls.trigger,
                current_frame=t1 if not is_new_page else None,
            )

            print(f"[step {i}] {cls.transition.value}  diff={cls.diff_ratio:.3f}  "
                  f"phash={cls.phash_distance}  crops={len(crops)}  → calling model...")
            t_model_start = time.time()
            resp = await model.predict(obs, state)
            t_model_s = time.time() - t_model_start

            est_tokens = 1600 if is_new_page else 400  # rough estimate
            (step_dir / "model.json").write_text(json.dumps({
                "obs_type": obs.obs_type,
                "trigger_reason": cls.trigger,
                "model_time_s": t_model_s,
                "action": str(resp.action) if resp.action else None,
                "done": resp.done,
                "reasoning": resp.reasoning[:500],
                "confidence": resp.confidence,
            }, indent=2))
            (step_dir / "classify.json").write_text(json.dumps({
                "step": i,
                "obs_type": obs.obs_type,
                "transition": cls.transition.value,
                "trigger": cls.trigger,
                "diff_ratio": cls.diff_ratio,
                "phash_distance": cls.phash_distance,
                "anchor_score": cls.anchor_score,
                "estimated_tokens": est_tokens,
                "num_crops": len(crops),
            }, indent=2))

            state.add_response(resp)
            state.log_transition(cls, resp0.action, i)

            if is_new_page:
                t0 = t1
                anchor = extract_anchor(t0, config)

            if resp.done or resp.action is None:
                print(f"  model returned done at step {i}")
                break

            resp0 = resp  # last_action for next obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--model", default="qwen2.5vl:7b")
    p.add_argument("--out", type=Path, default=Path("benchmarks/demo_run"))
    p.add_argument("--task", default=(
        "For each step: look at the screen and output "
        '{"type":"wait","duration_ms":300}. '
        "Stay on the current page. After 3 steps total, respond with done=true."
    ))
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args.steps, args.out, args.host, args.port, args.model, args.task))
    print(f"\nArtifacts saved to {args.out}/")


if __name__ == "__main__":
    main()
