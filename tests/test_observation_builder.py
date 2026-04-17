"""
Tests for observation/builder.py — the factory that produces typed
Observation objects for the model to consume.

Covered:
  - build_observation("full_frame", ...) returns FullFrameObservation with
    all fields correctly wired
  - build_observation("delta", ...) returns DeltaObservation
  - Defaults (empty crops, no text_deltas, etc.)
  - post_init correctly sets obs_type invariant
"""

import numpy as np
from PIL import Image

from agent.actions import Action, ActionType
from observation.builder import build_observation
from observation.types import FullFrameObservation, DeltaObservation


def _img(color=(128, 128, 128), size=(100, 100)):
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


class TestFullFrameBuilder:
    def test_basic_shape(self):
        frame = _img()
        obs = build_observation(
            obs_type="full_frame",
            task="Task X",
            step=5,
            last_action=None,
            frame=frame,
            url="https://example.com",
            trigger_reason="url_change",
        )
        assert isinstance(obs, FullFrameObservation)
        assert obs.obs_type == "full_frame"
        assert obs.task == "Task X"
        assert obs.step == 5
        assert obs.last_action is None
        assert obs.frame is frame
        assert obs.url == "https://example.com"
        assert obs.trigger_reason == "url_change"

    def test_with_last_action(self):
        action = Action(type=ActionType.CLICK, x=10, y=20)
        obs = build_observation(
            obs_type="full_frame",
            task="t", step=1,
            last_action=action,
            frame=_img(),
            url="",
            trigger_reason="initial",
        )
        assert obs.last_action is action

    def test_missing_optional_fields_use_defaults(self):
        """url and trigger_reason default to empty string."""
        obs = build_observation(
            obs_type="full_frame",
            task="t", step=0, last_action=None,
            frame=_img(),
        )
        assert obs.url == ""
        assert obs.trigger_reason == ""


class TestDeltaBuilder:
    def test_basic_shape(self):
        obs = build_observation(
            obs_type="delta",
            task="Task Y",
            step=3,
            last_action=Action(type=ActionType.TYPE, text="hi"),
            diff_result=None,
            crops=[{"bbox": (0, 0, 10, 10), "change_magnitude": 0.5}],
            action_had_effect=True,
            no_change_count=0,
        )
        assert isinstance(obs, DeltaObservation)
        assert obs.obs_type == "delta"
        assert obs.task == "Task Y"
        assert obs.step == 3
        assert obs.action_had_effect is True
        assert obs.no_change_count == 0
        assert len(obs.crops) == 1

    def test_empty_crops_default(self):
        """crops=None should yield an empty list, not fail."""
        obs = build_observation(
            obs_type="delta",
            task="t", step=0,
            last_action=None,
            diff_result=None,
        )
        assert obs.crops == []
        assert obs.text_deltas == []

    def test_text_deltas_preserved(self):
        td = [{"bbox": (0, 0, 50, 20), "before": "old", "after": "new"}]
        obs = build_observation(
            obs_type="delta",
            task="t", step=0,
            last_action=None,
            text_deltas=td,
        )
        assert obs.text_deltas == td

    def test_current_frame_preserved(self):
        frame = _img(color=(200, 100, 50))
        obs = build_observation(
            obs_type="delta",
            task="t", step=0,
            last_action=None,
            current_frame=frame,
        )
        assert obs.current_frame is frame


class TestDispatch:
    def test_unknown_type_falls_to_delta(self):
        """build_observation dispatches on obs_type string. Any non-'full_frame'
        value is treated as delta. This test documents that invariant."""
        obs = build_observation(
            obs_type="unknown_mode",
            task="t", step=0,
            last_action=None,
        )
        # Falls through to DeltaObservation branch, then __post_init__ normalizes
        assert isinstance(obs, DeltaObservation)
        assert obs.obs_type == "delta"  # post_init forces it


class TestPostInitInvariants:
    def test_full_frame_post_init_forces_type(self):
        """Even if caller bypasses builder and constructs directly,
        __post_init__ ensures obs_type is correct."""
        obs = FullFrameObservation(
            obs_type="WRONG",
            task="t", step=0, last_action=None,
            frame=_img(), url="", trigger_reason="",
        )
        assert obs.obs_type == "full_frame"

    def test_delta_post_init_forces_type(self):
        obs = DeltaObservation(
            obs_type="WRONG",
            task="t", step=0, last_action=None,
        )
        assert obs.obs_type == "delta"
