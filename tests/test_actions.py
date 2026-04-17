"""
Action parsing tests for V2's extended action space.

V2 adds DOUBLE_CLICK, RIGHT_CLICK, DRAG, HOTKEY to V1's 6 action types.
parse_action must handle all 10 types plus both DeltaVision-native and
UI-TARS / CogAgent formats, plus field coercion for local-VLM quirks.
"""

import pytest

from agent.actions import Action, ActionType, parse_action


class TestNativeFormat:
    def test_click(self):
        a = parse_action({"type": "click", "x": 100, "y": 200})
        assert a.type == ActionType.CLICK
        assert a.x == 100 and a.y == 200

    def test_double_click(self):
        a = parse_action({"type": "double_click", "x": 50, "y": 60})
        assert a.type == ActionType.DOUBLE_CLICK
        assert a.x == 50

    def test_right_click(self):
        a = parse_action({"type": "right_click", "x": 75, "y": 85})
        assert a.type == ActionType.RIGHT_CLICK

    def test_drag(self):
        a = parse_action({"type": "drag", "x": 10, "y": 20, "x2": 100, "y2": 200})
        assert a.type == ActionType.DRAG
        assert (a.x, a.y, a.x2, a.y2) == (10, 20, 100, 200)

    def test_hotkey(self):
        a = parse_action({"type": "hotkey", "key": "cmd+shift+3"})
        assert a.type == ActionType.HOTKEY
        assert a.key == "cmd+shift+3"

    def test_type(self):
        a = parse_action({"type": "type", "text": "hello world"})
        assert a.type == ActionType.TYPE
        assert a.text == "hello world"

    def test_scroll_down(self):
        a = parse_action({"type": "scroll", "direction": "down", "amount": 500})
        assert a.type == ActionType.SCROLL
        assert a.direction == "down"
        assert a.amount == 500

    def test_key(self):
        a = parse_action({"type": "key", "key": "Enter"})
        assert a.type == ActionType.KEY
        assert a.key == "Enter"

    def test_wait(self):
        a = parse_action({"type": "wait", "duration_ms": 2000})
        assert a.type == ActionType.WAIT
        assert a.duration_ms == 2000

    def test_done(self):
        a = parse_action({"type": "done"})
        assert a.type == ActionType.DONE


class TestUITarsFormat:
    def test_left_click_mapped_to_click(self):
        a = parse_action({"action": "left_click", "coordinate": [100, 200]})
        assert a.type == ActionType.CLICK
        assert (a.x, a.y) == (100, 200)

    def test_double_click_mapped(self):
        a = parse_action({"action": "double_click", "coordinate": [50, 60]})
        assert a.type == ActionType.DOUBLE_CLICK

    def test_right_click_mapped(self):
        a = parse_action({"action": "right_click", "coordinate": [10, 20]})
        assert a.type == ActionType.RIGHT_CLICK

    def test_drag_with_coordinate_end(self):
        a = parse_action({
            "action": "drag",
            "coordinate": [10, 20],
            "coordinate_end": [100, 200],
        })
        assert a.type == ActionType.DRAG
        assert (a.x, a.y, a.x2, a.y2) == (10, 20, 100, 200)

    def test_finished_mapped_to_done(self):
        a = parse_action({"action": "finished"})
        assert a.type == ActionType.DONE

    def test_press_mapped_to_key(self):
        a = parse_action({"action": "press", "key": "Enter"})
        assert a.type == ActionType.KEY

    def test_unknown_action_returns_none(self):
        """Unknown UI-TARS action → None (safer than guessing)."""
        a = parse_action({"action": "telepathy", "coordinate": [0, 0]})
        assert a is None


class TestCoercion:
    def test_string_coords_coerced(self):
        """Some local VLMs (MAI-UI-8B) emit coordinates as strings."""
        a = parse_action({"type": "click", "x": "551", "y": "300"})
        assert a.x == 551 and a.y == 300

    def test_invalid_coord_string_becomes_none(self):
        a = parse_action({"type": "click", "x": "not-a-number", "y": "300"})
        assert a.x is None and a.y == 300

    def test_float_coord_truncated(self):
        """Floats get int-cast; we don't care about fractional pixels."""
        a = parse_action({"type": "click", "x": 551.9, "y": 300.5})
        assert a.x == 551 and a.y == 300

    def test_drag_with_string_endpoints(self):
        a = parse_action({
            "type": "drag",
            "x": "10", "y": "20",
            "x2": "100", "y2": "200",
        })
        assert (a.x, a.y, a.x2, a.y2) == (10, 20, 100, 200)

    def test_amount_string_coerced(self):
        a = parse_action({"type": "scroll", "direction": "down", "amount": "300"})
        assert a.amount == 300

    def test_duration_ms_string_coerced(self):
        a = parse_action({"type": "wait", "duration_ms": "1500"})
        assert a.duration_ms == 1500


class TestEdgeCases:
    def test_none_input(self):
        assert parse_action(None) is None

    def test_empty_dict(self):
        assert parse_action({}) is None

    def test_string_input(self):
        assert parse_action("click 100 200") is None

    def test_unknown_type(self):
        """Unknown ActionType string → None."""
        assert parse_action({"type": "fly_to_mars"}) is None

    def test_missing_required_coords_still_returns_action(self):
        """Loop's responsibility to validate — parse just maps fields."""
        a = parse_action({"type": "click"})
        assert a.type == ActionType.CLICK
        assert a.x is None and a.y is None


class TestStringRepresentations:
    """Make sure __str__ doesn't crash for any action type — used in logs."""

    @pytest.mark.parametrize("action", [
        Action(type=ActionType.CLICK, x=10, y=20),
        Action(type=ActionType.DOUBLE_CLICK, x=30, y=40),
        Action(type=ActionType.RIGHT_CLICK, x=50, y=60),
        Action(type=ActionType.DRAG, x=10, y=10, x2=100, y2=100),
        Action(type=ActionType.TYPE, text="hello"),
        Action(type=ActionType.SCROLL, direction="down", amount=300),
        Action(type=ActionType.KEY, key="Enter"),
        Action(type=ActionType.HOTKEY, key="cmd+c"),
        Action(type=ActionType.WAIT, duration_ms=1000),
        Action(type=ActionType.DONE),
    ])
    def test_str_does_not_crash(self, action):
        s = str(action)
        assert isinstance(s, str)
        assert len(s) > 0
        # Type name should appear in the repr
        assert action.type.value in s.lower() or s == "done"
