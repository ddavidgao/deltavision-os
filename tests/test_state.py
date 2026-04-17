"""
Tests for agent/state.py — AgentState dataclass and its methods.

State is the agent's memory of what it's seen and done. Every method
that mutates state should have coverage so regressions don't silently
corrupt run logs or delta-ratio calculations.
"""

from dataclasses import dataclass
from enum import Enum

import pytest

from agent.state import AgentState


# Lightweight stand-ins for classification objects (no cv2 deps needed here)
class _T(Enum):
    DELTA = "delta"
    NEW_PAGE = "new_page"


@dataclass
class _Classification:
    transition: _T
    trigger: str
    diff_ratio: float
    phash_distance: int
    anchor_score: float


class TestInitialState:
    def test_defaults(self):
        s = AgentState(task="test task")
        assert s.task == "test task"
        assert s.step == 0
        assert s.done is False
        assert s.no_change_streak == 0
        assert s.new_page_count == 0
        assert s.observations == []
        assert s.responses == []
        assert s.transition_log == []

    def test_delta_ratio_empty_is_zero(self):
        s = AgentState(task="x")
        assert s.delta_ratio == 0.0


class TestNoChangeStreak:
    def test_increment(self):
        s = AgentState(task="x")
        s.increment_no_change_streak()
        s.increment_no_change_streak()
        assert s.no_change_streak == 2

    def test_reset(self):
        s = AgentState(task="x")
        s.increment_no_change_streak()
        s.increment_no_change_streak()
        s.reset_no_change_streak()
        assert s.no_change_streak == 0

    def test_reset_from_zero_noop(self):
        s = AgentState(task="x")
        s.reset_no_change_streak()
        assert s.no_change_streak == 0


class TestNewPageCount:
    def test_increment(self):
        s = AgentState(task="x")
        s.increment_new_page_count()
        s.increment_new_page_count()
        s.increment_new_page_count()
        assert s.new_page_count == 3


class TestTransitionLog:
    def test_log_transition_basic(self):
        s = AgentState(task="x")
        cls = _Classification(
            transition=_T.DELTA, trigger="none",
            diff_ratio=0.03, phash_distance=2, anchor_score=0.98,
        )
        s.log_transition(cls, action="click(10,20)", step=1)
        assert len(s.transition_log) == 1
        entry = s.transition_log[0]
        assert entry["step"] == 1
        assert entry["action"] == "click(10,20)"
        assert entry["transition"] == "delta"
        assert entry["trigger"] == "none"
        assert entry["diff_ratio"] == 0.03

    def test_log_multiple(self):
        s = AgentState(task="x")
        for i in range(5):
            cls = _Classification(
                transition=_T.DELTA if i % 2 == 0 else _T.NEW_PAGE,
                trigger="test", diff_ratio=0.1, phash_distance=5, anchor_score=0.9,
            )
            s.log_transition(cls, action=f"step-{i}", step=i)
        assert len(s.transition_log) == 5
        assert [t["step"] for t in s.transition_log] == [0, 1, 2, 3, 4]


class TestDeltaRatio:
    def test_all_delta_returns_one(self):
        s = AgentState(task="x")
        for _ in range(3):
            s.log_transition(
                _Classification(
                    transition=_T.DELTA, trigger="none",
                    diff_ratio=0.0, phash_distance=0, anchor_score=1.0,
                ),
                action="x", step=0,
            )
        assert s.delta_ratio == 1.0

    def test_all_new_page_returns_zero(self):
        s = AgentState(task="x")
        for _ in range(3):
            s.log_transition(
                _Classification(
                    transition=_T.NEW_PAGE, trigger="url",
                    diff_ratio=0.9, phash_distance=30, anchor_score=0.3,
                ),
                action="x", step=0,
            )
        assert s.delta_ratio == 0.0

    def test_mixed_ratio(self):
        s = AgentState(task="x")
        # 3 delta, 1 new_page → 0.75
        for t in [_T.DELTA, _T.DELTA, _T.NEW_PAGE, _T.DELTA]:
            s.log_transition(
                _Classification(
                    transition=t, trigger="x",
                    diff_ratio=0.1, phash_distance=5, anchor_score=0.9,
                ),
                action="x", step=0,
            )
        assert s.delta_ratio == 0.75


class TestObservationResponseLogging:
    def test_add_observation(self):
        s = AgentState(task="x")
        s.add_observation("obs_0")
        s.add_observation("obs_1")
        assert s.observations == ["obs_0", "obs_1"]

    def test_add_response(self):
        s = AgentState(task="x")
        s.add_response({"a": 1})
        s.add_response({"b": 2})
        assert len(s.responses) == 2
        assert s.responses[0] == {"a": 1}
