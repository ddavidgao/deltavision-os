"""
Typed actions for DeltaVision-OS.

Extends V1's 6-action browser space with OS-level primitives: DOUBLE_CLICK,
RIGHT_CLICK, DRAG (with x2/y2), HOTKEY (multi-key combo).

Actions are always typed — no free-form strings. The platform (OSNative,
OSWorld, etc.) translates them into platform-specific events.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    # V1 (carried over)
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY = "key"
    WAIT = "wait"
    DONE = "done"

    # V2 (OS-level additions)
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"       # needs x, y (start) + x2, y2 (end)
    HOTKEY = "hotkey"   # multi-key combo: "ctrl+c", "cmd+shift+3"


@dataclass
class Action:
    type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    x2: Optional[int] = None   # V2: drag end x
    y2: Optional[int] = None   # V2: drag end y
    text: Optional[str] = None
    direction: Optional[str] = None
    amount: Optional[int] = None
    key: Optional[str] = None
    duration_ms: Optional[int] = None

    def __str__(self):
        match self.type:
            case ActionType.CLICK:
                return f"click({self.x}, {self.y})"
            case ActionType.DOUBLE_CLICK:
                return f"double_click({self.x}, {self.y})"
            case ActionType.RIGHT_CLICK:
                return f"right_click({self.x}, {self.y})"
            case ActionType.DRAG:
                return f"drag({self.x},{self.y} -> {self.x2},{self.y2})"
            case ActionType.TYPE:
                return f"type('{self.text}')"
            case ActionType.SCROLL:
                return f"scroll({self.direction}, {self.amount}px)"
            case ActionType.KEY:
                return f"key({self.key})"
            case ActionType.HOTKEY:
                return f"hotkey({self.key})"
            case ActionType.WAIT:
                return f"wait({self.duration_ms}ms)"
            case ActionType.DONE:
                return "done"
        return f"unknown({self.type})"


def parse_action(action_dict: Optional[dict]) -> Optional[Action]:
    """Parse model JSON output into a typed Action.

    Accepts two formats:
      1. DeltaVision native: {"type": "click", "x": 100, "y": 200}
      2. UI-TARS / CogAgent: {"action": "left_click", "coordinate": [100, 200]}
    """
    if not action_dict or isinstance(action_dict, str):
        return None

    try:
        # UI-TARS / CogAgent format
        if "action" in action_dict and "type" not in action_dict:
            raw = action_dict["action"]
            coord = action_dict.get("coordinate", [])

            action_map = {
                "left_click": ActionType.CLICK,
                "click": ActionType.CLICK,
                "right_click": ActionType.RIGHT_CLICK,
                "double_click": ActionType.DOUBLE_CLICK,
                "drag": ActionType.DRAG,
                "type": ActionType.TYPE,
                "scroll": ActionType.SCROLL,
                "key": ActionType.KEY,
                "press": ActionType.KEY,
                "hotkey": ActionType.HOTKEY,
                "wait": ActionType.WAIT,
                "finished": ActionType.DONE,
                "done": ActionType.DONE,
            }
            atype = action_map.get(raw.lower())
            if atype is None:
                return None

            coord2 = action_dict.get("coordinate_end", [])
            return Action(
                type=atype,
                x=int(coord[0]) if len(coord) > 0 else None,
                y=int(coord[1]) if len(coord) > 1 else None,
                x2=int(coord2[0]) if len(coord2) > 0 else None,
                y2=int(coord2[1]) if len(coord2) > 1 else None,
                text=action_dict.get("text"),
                direction=action_dict.get("direction"),
                amount=action_dict.get("amount"),
                key=action_dict.get("key"),
            )

        # DeltaVision native format. Coerce numeric fields — some VLMs emit
        # coordinates as strings.
        def _int(v):
            if v is None:
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        return Action(
            type=ActionType(action_dict["type"]),
            x=_int(action_dict.get("x")),
            y=_int(action_dict.get("y")),
            x2=_int(action_dict.get("x2")),
            y2=_int(action_dict.get("y2")),
            text=action_dict.get("text"),
            direction=action_dict.get("direction"),
            amount=_int(action_dict.get("amount")),
            key=action_dict.get("key"),
            duration_ms=_int(action_dict.get("duration_ms")),
        )
    except (KeyError, ValueError, TypeError, IndexError):
        return None
