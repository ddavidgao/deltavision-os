"""
ScreenSpot-v2 evaluation harness.

Measures VLM grounding accuracy on 1,272 UI screenshots (desktop/mobile/web,
text/icon targets). Complements the token-savings ablation — they measure
orthogonal things (CV pipeline efficiency vs. model grounding quality).

Dataset: OS-Copilot/ScreenSpot-v2 on HuggingFace. Format is 3 JSON files
(desktop/mobile/web) + a zip of PNGs. Not a standard HF parquet.

Scoring: a prediction is correct if the predicted (x,y) click point falls
inside the ground-truth bbox. Same convention as SeeClick/OS-Atlas/UI-TARS
papers.

Runtime: ~3-5s per example via Ollama on the 5080 box. Sample of 30 runs
in ~2 min; full 1272 in ~80 min.

Usage:
    # Quick sanity: 10 examples per platform (30 total)
    python benchmarks/screenspot_eval.py --model qwen2.5vl:7b

    # Head-to-head: two models, subset
    python benchmarks/screenspot_eval.py --model qwen2.5vl:7b
    python benchmarks/screenspot_eval.py --model 0000/ui-tars-1.5-7b:latest --adapter ui-tars

    # Full run
    python benchmarks/screenspot_eval.py --model qwen2.5vl:7b --per-platform 0
"""

import argparse
import base64
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download
from PIL import Image
import openai


DATASET_REPO = "OS-Copilot/ScreenSpot-v2"
PLATFORMS = ("desktop", "mobile", "web")

PROMPT_SEECLICK = (
    'In this UI screenshot, what is the position of the element '
    'corresponding to the command "{instruction}" (with point)? '
    'Respond with JSON only: {{"point_2d": [x, y]}} in absolute pixel coordinates.'
)

PROMPT_UI_TARS = (
    "You are a GUI agent. Output a single click action to complete the instruction.\n"
    'Instruction: {instruction}\n'
    'Format: click(point=\'<point>x y</point>\')'
)


def ensure_dataset(cache_dir: Path) -> Path:
    return Path(snapshot_download(repo_id=DATASET_REPO, repo_type="dataset",
                                  cache_dir=str(cache_dir)))


def load_examples(dataset_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for p in PLATFORMS:
        data = json.loads((dataset_dir / f"screenspot_{p}_v2.json").read_text())
        out[p] = data
    return out


def open_image_zip(dataset_dir: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(dataset_dir / "screenspotv2_image.zip", "r")


def load_image(zf: zipfile.ZipFile, filename: str) -> Image.Image:
    # The zip typically has entries like "screenspotv2_image/pc_XXX.png"
    for name in [filename, f"screenspotv2_image/{filename}"]:
        try:
            with zf.open(name) as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        except KeyError:
            continue
    raise KeyError(f"{filename} not found in zip")


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def extract_point(text: str, image_size: tuple[int, int]) -> Optional[tuple[float, float]]:
    """Parse predicted click (x,y) from raw model output.

    Handles: JSON point/bbox, UI-TARS <point>, raw number pairs, 0-1 normalized,
    0-1000 normalized. Returns coords in absolute image pixels.
    """
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).strip().strip("`")
    W, H = image_size

    # Try JSON
    for candidate in [text] + re.findall(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("point_2d", "point", "click_point", "position"):
            p = obj.get(key)
            if p and len(p) >= 2:
                return _maybe_unnorm(float(p[0]), float(p[1]), W, H)
        for key in ("bbox_2d", "bbox", "box"):
            b = obj.get(key)
            if b and len(b) >= 4:
                x1, y1, x2, y2 = [float(v) for v in b[:4]]
                return _maybe_unnorm((x1 + x2) / 2, (y1 + y2) / 2, W, H)

    # UI-TARS / CogAgent style: <point>x y</point> or <box>x1 y1 x2 y2</box>
    m = re.search(r"<point>\s*([\d.]+)[\s,]+([\d.]+)\s*</point>", text)
    if m:
        return _maybe_unnorm(float(m.group(1)), float(m.group(2)), W, H)
    m = re.search(r"<box>\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)\s*</box>", text)
    if m:
        x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
        return _maybe_unnorm((x1 + x2) / 2, (y1 + y2) / 2, W, H)

    # start_box='(x1,y1,x2,y2)' — UI-TARS action schema
    m = re.search(r"start_box\s*=\s*['\"]?\(?\s*([\d.]+)[\s,]+([\d.]+)"
                  r"(?:[\s,]+([\d.]+)[\s,]+([\d.]+))?\s*\)?['\"]?", text)
    if m:
        xs = [float(g) for g in m.groups() if g is not None]
        if len(xs) == 4:
            return _maybe_unnorm((xs[0] + xs[2]) / 2, (xs[1] + xs[3]) / 2, W, H)
        return _maybe_unnorm(xs[0], xs[1], W, H)

    # Last-resort: first two numbers
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) >= 2:
        try:
            return _maybe_unnorm(float(nums[0]), float(nums[1]), W, H)
        except ValueError:
            pass
    return None


def _maybe_unnorm(x: float, y: float, W: int, H: int) -> tuple[float, float]:
    # If both look like 0-1 normalized
    if 0 <= x <= 1.0 and 0 <= y <= 1.0:
        return (x * W, y * H)
    # If both look like 0-1000 normalized (and out of image bounds)
    if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > W or y > H):
        return (x * W / 1000, y * H / 1000)
    return (x, y)


def is_hit(point: tuple[float, float], bbox_xywh: list[int]) -> bool:
    x, y = point
    bx, by, bw, bh = bbox_xywh
    return bx <= x <= bx + bw and by <= y <= by + bh


def build_prompt(adapter: str, instruction: str) -> str:
    if adapter == "ui-tars":
        return PROMPT_UI_TARS.format(instruction=instruction)
    return PROMPT_SEECLICK.format(instruction=instruction)


def predict(client, model: str, image: Image.Image, instruction: str,
            adapter: str, max_tokens: int = 128) -> str:
    prompt = build_prompt(adapter, instruction)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def run_eval(args) -> dict:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: downloading/loading {DATASET_REPO} to {cache_dir}")
    dataset_dir = ensure_dataset(cache_dir)
    examples_by_platform = load_examples(dataset_dir)
    zf = open_image_zip(dataset_dir)

    if args.per_platform > 0:
        for p in PLATFORMS:
            examples_by_platform[p] = examples_by_platform[p][:args.per_platform]
    total = sum(len(v) for v in examples_by_platform.values())

    print(f"Model: {args.model} (adapter={args.adapter}) via {args.base_url}")
    print(f"Evaluating {total} examples "
          f"(desktop={len(examples_by_platform['desktop'])}, "
          f"mobile={len(examples_by_platform['mobile'])}, "
          f"web={len(examples_by_platform['web'])})\n")

    client = openai.OpenAI(api_key="not-needed", base_url=args.base_url)

    results = []
    t0 = time.time()
    done = 0

    for platform, examples in examples_by_platform.items():
        for ex in examples:
            done += 1
            instr = ex["instruction"]
            bbox_xywh = ex["bbox"]
            ui_type = ex.get("data_type", "")

            try:
                img = load_image(zf, ex["img_filename"])
            except KeyError as e:
                print(f"  [{done:4d}/{total}] SKIP  {ex['img_filename']}: {e}")
                continue

            try:
                raw = predict(client, args.model, img, instr, args.adapter)
            except Exception as e:
                print(f"  [{done:4d}/{total}] ERR   {type(e).__name__}: "
                      f"{str(e)[:100]}")
                continue

            point = extract_point(raw, img.size)
            hit = is_hit(point, bbox_xywh) if point else False

            results.append({
                "platform": platform,
                "ui_type": ui_type,
                "instruction": instr,
                "img": ex["img_filename"],
                "img_size": list(img.size),
                "bbox_xywh": bbox_xywh,
                "predicted_point": list(point) if point else None,
                "hit": hit,
                "raw_response": raw[:300],
            })

            mark = "✓" if hit else ("·" if point else "?")
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                hits_so_far = sum(1 for r in results if r["hit"])
                acc = hits_so_far / len(results) * 100
                print(f"  [{done:4d}/{total}] {mark} {platform:<7s}{ui_type:<5s} "
                      f"acc_running={acc:5.1f}%  elapsed={elapsed:.0f}s")
            else:
                print(f"  [{done:4d}/{total}] {mark} {platform:<7s}{ui_type:<5s} "
                      f"{instr[:60]}")

    zf.close()
    return summarize(results, args, time.time() - t0)


def summarize(results: list[dict], args, elapsed: float) -> dict:
    by_pt = {}
    for r in results:
        k = (r["platform"], r["ui_type"])
        by_pt.setdefault(k, []).append(r["hit"])

    print("\n=== Results ===")
    print(f"{'platform':<10s}{'ui_type':<8s}{'n':>5s}{'accuracy':>12s}")
    print("-" * 35)
    overall = []
    per_platform_acc = {}
    for p in PLATFORMS:
        p_total, p_hits = 0, 0
        for ui in ("text", "icon"):
            items = by_pt.get((p, ui), [])
            if not items:
                continue
            n = len(items)
            acc = sum(items) / n * 100
            print(f"{p:<10s}{ui:<8s}{n:>5d}{acc:>11.1f}%")
            p_total += n
            p_hits += sum(items)
            overall.extend(items)
        if p_total:
            per_platform_acc[p] = p_hits / p_total * 100
            print(f"{p:<10s}{'-- all':<8s}{p_total:>5d}"
                  f"{per_platform_acc[p]:>11.1f}%")
    if overall:
        overall_acc = sum(overall) / len(overall) * 100
        print(f"\n{'OVERALL':<18s}{len(overall):>5d}{overall_acc:>11.1f}%")
    else:
        overall_acc = 0.0

    print(f"\nWall time: {elapsed:.0f}s  ({len(results)} scored)")

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "base_url": args.base_url,
        "per_platform_limit": args.per_platform,
        "total_scored": len(results),
        "overall_accuracy_pct": round(overall_acc, 2),
        "per_platform_accuracy_pct": {k: round(v, 2)
                                      for k, v in per_platform_acc.items()},
        "by_ui_type": {
            f"{p}_{ui}": {
                "n": len(by_pt.get((p, ui), [])),
                "accuracy_pct": round(
                    sum(by_pt.get((p, ui), [])) / max(len(by_pt.get((p, ui), [])), 1)
                    * 100, 2),
            }
            for p in PLATFORMS for ui in ("text", "icon")
        },
        "wall_time_s": round(elapsed, 1),
        "results": results,
    }

    out = Path(args.output) if args.output else \
        Path(__file__).parent / f"screenspot_result_{_slug(args.model)}.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    try:
        print(f"Artifact: {out.relative_to(Path.cwd())}")
    except ValueError:
        print(f"Artifact: {out}")
    return summary


def _slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5vl:7b",
                    help="Model tag (Ollama: qwen2.5vl:7b, 0000/ui-tars-1.5-7b:latest)")
    ap.add_argument("--adapter", default="seeclick", choices=("seeclick", "ui-tars"),
                    help="Prompt template + expected output format")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1",
                    help="OpenAI-compat endpoint (Ollama default shown)")
    ap.add_argument("--per-platform", type=int, default=10,
                    help="N examples per platform (0 = all 1272)")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "huggingface"),
                    help="HF cache root")
    ap.add_argument("--output", default=None, help="Write JSON here")
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
