"""
Hybrid test: real mss screen capture on Mac + scripted model + pipeline.

This is the closest thing to "run it for real" without needing a VLM or
actually driving the cursor. We capture two real screenshots 100ms apart,
feed them through the CV pipeline via a wrapper platform that returns
our captures but executes no-op actions.

Marks: skipped in headless CI — needs a real display.
"""

import asyncio
from typing import List

import pytest
from PIL import Image

from agent.actions import Action, ActionType
from agent.loop import run_agent
from capture.base import Platform
from capture.os_native import OSNativePlatform
from config import DeltaVisionConfig
from model.scripted import ScriptedModel


class RealCaptureNoopExecute(Platform):
    """Real mss capture on setup/capture, no-op execute.

    Lets us observe the CV pipeline running against genuine desktop frames
    without actually clicking anything. The mss instance is held by an
    inner OSNativePlatform.
    """

    def __init__(self):
        self._inner = OSNativePlatform(cursor_park=None)
        self.actions_attempted: List[Action] = []

    async def setup(self):
        await self._inner.setup()

    async def capture(self) -> Image.Image:
        return await self._inner.capture()

    async def get_url(self):
        return None

    async def execute(self, action: Action):
        # Record but do NOT actually execute — no keyboard/mouse events sent.
        self.actions_attempted.append(action)

    async def teardown(self):
        await self._inner.teardown()


class TestRealCaptureLoop:
    def test_pipeline_runs_against_real_frames(self):
        """Loop completes without crashing on real captured desktop frames."""
        platform = RealCaptureNoopExecute()
        config = DeltaVisionConfig(MAX_STEPS=3, POST_ACTION_WAIT_MS=100)
        # Scripted: 2 no-op wait actions to trigger 2 capture cycles
        model = ScriptedModel([
            Action(type=ActionType.WAIT, duration_ms=100),
            Action(type=ActionType.WAIT, duration_ms=100),
        ])

        async def run():
            async with platform:
                return await run_agent(
                    task="observe desktop twice",
                    model=model,
                    platform=platform,
                    config=config,
                )

        state = asyncio.run(run())
        assert state.step == 2
        assert len(state.transition_log) == 2
        # Each transition must have measured real CV metrics
        for t in state.transition_log:
            assert "diff_ratio" in t
            assert 0.0 <= t["diff_ratio"] <= 1.0
            assert 0 <= t["phash_distance"] <= 64

    def test_quiet_desktop_classifies_as_delta(self):
        """Back-to-back captures on a quiet desktop should NOT trigger
        NEW_PAGE — should be DELTA (or even zero-change DELTA)."""
        platform = RealCaptureNoopExecute()
        config = DeltaVisionConfig(MAX_STEPS=3, POST_ACTION_WAIT_MS=50)
        model = ScriptedModel([Action(type=ActionType.WAIT, duration_ms=50)])

        async def run():
            async with platform:
                return await run_agent(
                    task="quiet observation",
                    model=model,
                    platform=platform,
                    config=config,
                )

        state = asyncio.run(run())
        # The single transition should be DELTA (quiet desktop, tiny diff)
        assert len(state.transition_log) == 1
        t = state.transition_log[0]
        # Diff ratio should be small — not a new page
        assert t["diff_ratio"] < 0.30, \
            f"Expected quiet desktop diff < 30%, got {t['diff_ratio']:.1%}"
        assert t["transition"] == "delta"
