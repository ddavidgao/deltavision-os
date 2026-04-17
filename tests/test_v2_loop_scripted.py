"""
End-to-end test: V2 agent loop runs through the full pipeline against
a MockPlatform (no real OS actions) with a ScriptedModel.

Proves that the platform abstraction, CV pipeline, observation builder,
and agent state all wire together correctly without needing a real VLM
or real desktop.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent.actions import Action, ActionType
from agent.loop import run_agent
from capture.base import Platform
from config import DeltaVisionConfig
from model.scripted import ScriptedModel


class MockPlatform(Platform):
    """
    In-memory platform. Returns a programmable sequence of frames.
    No real screen capture, no real actions. Pure test scaffolding.
    """

    def __init__(self, frames: list[Image.Image]):
        self.frames = frames
        self.current = 0
        self.actions_executed: list[Action] = []

    async def setup(self):
        pass

    async def capture(self) -> Image.Image:
        idx = min(self.current, len(self.frames) - 1)
        return self.frames[idx]

    async def get_url(self):
        return None

    async def execute(self, action: Action) -> None:
        self.actions_executed.append(action)
        # Advance to next frame to simulate a change
        self.current = min(self.current + 1, len(self.frames) - 1)

    async def teardown(self):
        pass


def _solid(color: tuple[int, int, int], size=(400, 300)) -> Image.Image:
    """Solid-color test frame."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:] = color
    return Image.fromarray(arr)


def _striped(variant: int, size=(400, 300)) -> Image.Image:
    """Test frame that changes materially between variants."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = ((x + variant * 30) % 255, (y + variant * 50) % 255, 128)
    return Image.fromarray(arr)


class TestLoopIntegration:
    def test_empty_script_terminates_after_initial_capture(self):
        """Model returns done immediately — loop should exit cleanly."""
        frames = [_solid((128, 128, 128))]
        platform = MockPlatform(frames)
        config = DeltaVisionConfig(MAX_STEPS=5)
        model = ScriptedModel([])  # no actions

        async def run():
            async with platform:
                return await run_agent(
                    task="nothing to do",
                    model=model,
                    platform=platform,
                    config=config,
                )

        state = asyncio.run(run())
        assert state.done
        assert state.step == 0
        assert len(state.observations) == 1   # initial full frame only

    def test_single_click_action_triggers_classification(self):
        """One click, platform swaps frame, classifier runs, observation logged."""
        frames = [_striped(0), _striped(5)]  # materially different
        platform = MockPlatform(frames)
        config = DeltaVisionConfig(MAX_STEPS=5)
        model = ScriptedModel([Action(type=ActionType.CLICK, x=100, y=100)])

        async def run():
            async with platform:
                return await run_agent(
                    task="click something",
                    model=model,
                    platform=platform,
                    config=config,
                )

        state = asyncio.run(run())
        assert state.step == 1
        assert len(platform.actions_executed) == 1
        assert platform.actions_executed[0].type == ActionType.CLICK
        assert len(state.transition_log) == 1

    def test_no_change_streak_forces_full_refresh(self):
        """Multiple no-effect actions trigger force_refresh_no_effect."""
        # Frames stay identical — every action has no effect
        frames = [_solid((80, 80, 80))]
        platform = MockPlatform(frames)
        config = DeltaVisionConfig(MAX_STEPS=10, MAX_NO_EFFECT_RETRIES=2)
        model = ScriptedModel([
            Action(type=ActionType.CLICK, x=50, y=50),
            Action(type=ActionType.CLICK, x=60, y=60),
            Action(type=ActionType.CLICK, x=70, y=70),
            Action(type=ActionType.CLICK, x=80, y=80),
        ])

        async def run():
            async with platform:
                return await run_agent(
                    task="stuck actions",
                    model=model,
                    platform=platform,
                    config=config,
                )

        state = asyncio.run(run())
        # Some observation should have trigger_reason containing force_refresh
        triggers = [
            getattr(o, "trigger_reason", "")
            for o in state.observations
        ]
        assert any("force_refresh_no_effect" in t for t in triggers), \
            f"Expected force_refresh trigger, got: {triggers}"

    def test_platform_get_url_none_does_not_crash_loop(self):
        """Key test: OS-native platforms return URL=None, classifier must
        handle gracefully without crashing on string comparisons."""
        frames = [_striped(0), _striped(10), _striped(20)]
        platform = MockPlatform(frames)
        config = DeltaVisionConfig(MAX_STEPS=5)
        model = ScriptedModel([
            Action(type=ActionType.CLICK, x=100, y=100),
            Action(type=ActionType.CLICK, x=200, y=200),
        ])

        async def run():
            async with platform:
                return await run_agent(
                    task="url-less platform",
                    model=model,
                    platform=platform,
                    config=config,
                )

        state = asyncio.run(run())
        # Should complete without raising
        assert state.step == 2
        assert len(state.transition_log) == 2

    def test_drag_action_type_roundtrips(self):
        """V2-specific action type should parse, execute, and log correctly."""
        frames = [_striped(0), _striped(8)]
        platform = MockPlatform(frames)
        config = DeltaVisionConfig(MAX_STEPS=5)
        model = ScriptedModel([
            Action(type=ActionType.DRAG, x=10, y=10, x2=200, y2=150)
        ])

        async def run():
            async with platform:
                return await run_agent(
                    task="drag test",
                    model=model,
                    platform=platform,
                    config=config,
                )

        state = asyncio.run(run())
        assert len(platform.actions_executed) == 1
        assert platform.actions_executed[0].type == ActionType.DRAG
        # State log uses str(action) which should match the V2 drag format
        assert "drag(" in state.transition_log[0]["action"]
