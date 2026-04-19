"""
DeltaVision-OS CLI — delta-first OS-level computer use agent.

Unlike V1 (which drove a Playwright browser), V2 drives the real desktop
via mss (capture) and pyautogui (actions). The CV pipeline is identical.

Usage:
    # Live desktop observation with scripted no-op actions (safe demo)
    python main.py --task "observe desktop" --platform os --model scripted --max-steps 5

    # Against a llama.cpp server (when running)
    python main.py --task "..." --platform os --backend llamacpp \\
        --host 100.70.57.66 --port 8080 --model qwen3-vl-8b

    # Claude API (for testing the V2 stack against a known-good model)
    python main.py --task "..." --platform os --backend claude

    # OSWorld VM (when env is wired up)
    python main.py --task "..." --platform osworld --task-id os.browser.chrome.001

IMPORTANT: --platform os drives the REAL DESKTOP. The model will control
your mouse and keyboard. Start with --safety strict and --max-steps small.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    for _d in [Path.cwd(), Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        _env = _d / ".env"
        if _env.exists():
            load_dotenv(_env, override=True)
            break
except ImportError:
    pass  # optional

from deltavision_os.config import DeltaVisionConfig
from deltavision_os.agent.loop import run_agent


def build_platform(args):
    """Construct the concrete Platform instance per --platform flag."""
    if args.platform == "os":
        from deltavision_os.capture.os_native import OSNativePlatform
        return OSNativePlatform(
            monitor=args.monitor,
            cursor_park=(args.cursor_park_x, args.cursor_park_y)
            if args.cursor_park_x is not None else None,
        )
    elif args.platform == "osworld":
        from deltavision_os.capture.osworld import OSWorldPlatform
        # OSWorld env instance must be passed via env hook; this is a stub
        return OSWorldPlatform(task_id=args.task_id)
    else:
        print(f"Unknown platform: {args.platform}", file=sys.stderr)
        sys.exit(1)


def build_model(args, config):
    """Construct the model backend per --backend flag."""
    if args.backend == "scripted":
        from deltavision_os.agent.actions import Action, ActionType
        from deltavision_os.model.scripted import ScriptedModel
        # Default: N no-op WAIT actions so the pipeline runs but nothing executes
        actions = [
            Action(type=ActionType.WAIT, duration_ms=500)
            for _ in range(args.max_steps)
        ]
        return ScriptedModel(actions)

    if args.backend == "llamacpp":
        from deltavision_os.model.llamacpp import LlamaCppModel
        return LlamaCppModel(
            host=args.host,
            port=args.port,
            model=args.model or "default",
        )

    if args.backend == "claude":
        from deltavision_os.model.claude import ClaudeModel
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        return ClaudeModel(api_key=key, model=args.model or "claude-sonnet-4-6")

    if args.backend == "openai":
        from deltavision_os.model.openai import OpenAIModel
        key = os.environ.get("OPENAI_API_KEY") or "sk-no-key"
        return OpenAIModel(
            api_key=key,
            model=args.model or "gpt-4o",
            base_url=args.base_url,
        )

    if args.backend == "ollama":
        from deltavision_os.model.ollama import OllamaModel
        return OllamaModel(
            model=args.model or "qwen2.5vl:7b",
            host=f"http://{args.host}:{args.port or 11434}",
        )

    print(f"Unknown backend: {args.backend}", file=sys.stderr)
    sys.exit(1)


def build_safety(mode):
    if mode == "none":
        return None
    from deltavision_os.safety import PERMISSIVE, STRICT, EDUCATIONAL
    return {"permissive": PERMISSIVE, "strict": STRICT, "educational": EDUCATIONAL}[mode]


async def main(args):
    config = DeltaVisionConfig(MAX_STEPS=args.max_steps)
    if args.force_full_frame:
        config.FORCE_FULL_FRAME = True

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    platform = build_platform(args)
    model = build_model(args, config)
    safety = build_safety(args.safety)

    # Safety warning for OS platform
    if args.platform == "os" and args.backend != "scripted":
        print(">> WARNING: --platform os will drive your REAL mouse and keyboard.")
        print(f">> Max steps: {args.max_steps}.  Safety: {args.safety}.")
        print(">> Ctrl+C to abort within the first second.")
        import time
        time.sleep(2.0)

    async with platform:
        state = await run_agent(
            task=args.task,
            model=model,
            platform=platform,
            config=config,
            safety=safety,
        )

    # Dump results
    result = {
        "task": state.task,
        "platform": args.platform,
        "backend": args.backend,
        "steps": state.step,
        "done": state.done,
        "delta_ratio": round(state.delta_ratio, 3),
        "new_page_count": state.new_page_count,
        "transition_log": state.transition_log,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = args.output or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Steps: {state.step}, Delta ratio: {state.delta_ratio:.1%}, "
          f"New pages: {state.new_page_count}")


def _build_parser():
    p = argparse.ArgumentParser(description="DeltaVision-OS — OS-level delta-first agent")
    p.add_argument("--task", required=True, help="Task description")
    p.add_argument("--platform", choices=["os", "osworld"], default="os")
    p.add_argument("--backend", choices=["scripted", "llamacpp", "claude", "openai", "ollama"],
                   default="scripted", help="Model backend")
    p.add_argument("--model", help="Model name/ID override")
    p.add_argument("--host", default="localhost", help="Model server host")
    p.add_argument("--port", type=int, help="Model server port")
    p.add_argument("--base-url", help="OpenAI-compatible base URL")
    p.add_argument("--task-id", help="OSWorld task ID (only for --platform osworld)")
    p.add_argument("--monitor", type=int, default=1, help="mss monitor index (1=primary)")
    p.add_argument("--cursor-park-x", type=int, help="Park cursor at this X before capture")
    p.add_argument("--cursor-park-y", type=int)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--safety", choices=["none", "permissive", "strict", "educational"],
                   default="permissive")
    p.add_argument("--force-full-frame", action="store_true",
                   help="Ablation: always send full frame, disable delta gating")
    p.add_argument("--output", help="Output JSON path")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def cli_entry():
    args = _build_parser().parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_entry()
