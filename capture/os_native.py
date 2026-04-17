"""
OS-native platform: mss for capture, pyautogui for execution.

Works on macOS, Linux, Windows. Captures the primary monitor by default.
pyautogui drives the real cursor and keyboard — there is no sandbox here.
Make sure you trust the model before pointing it at your desktop.
"""

import asyncio
from typing import Optional

from PIL import Image
import mss
import mss.tools

from capture.base import Platform


class OSNativePlatform(Platform):
    """Primary-monitor capture via mss, mouse/keyboard via pyautogui."""

    def __init__(self, monitor: int = 1, cursor_park: Optional[tuple[int, int]] = None):
        """
        Args:
            monitor: mss monitor index. 0 = all monitors combined, 1 = primary.
                     Use 1 unless you explicitly want multi-monitor capture.
            cursor_park: (x, y) to move cursor before every capture, so the
                         cursor doesn't pollute the diff. Set to None to skip.
        """
        self.monitor_idx = monitor
        self.cursor_park = cursor_park
        self._mss: Optional[mss.base.MSSBase] = None
        self._pyautogui = None

    async def setup(self) -> None:
        self._mss = mss.mss()
        # Lazy import: pyautogui touches the display at import time, which
        # can fail on headless Linux. Only import when we actually use it.
        import pyautogui
        pyautogui.FAILSAFE = True   # moving cursor to corner aborts — safety net
        pyautogui.PAUSE = 0          # we manage our own pacing
        self._pyautogui = pyautogui

    async def capture(self) -> Image.Image:
        if self._mss is None:
            raise RuntimeError("Platform not set up — call setup() first")

        if self.cursor_park is not None and self._pyautogui is not None:
            # Park the cursor so it doesn't appear in the diff between frames.
            # Small compromise: the model never sees cursor location.
            self._pyautogui.moveTo(*self.cursor_park, _pause=False)
            await asyncio.sleep(0.02)

        sct = self._mss.grab(self._mss.monitors[self.monitor_idx])
        # mss returns BGRA; PIL needs RGB.
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        return img

    async def get_url(self) -> Optional[str]:
        # Desktop has no URL. The classifier will use diff/pHash/anchor
        # to detect NEW_PAGE-equivalent transitions (e.g. app switch).
        return None

    async def execute(self, action) -> None:
        if self._pyautogui is None:
            raise RuntimeError("Platform not set up — call setup() first")

        # Import here to avoid a circular-ish dep in the scaffold.
        # Once agent/actions.py lands, this will be a direct import.
        from agent.actions import ActionType

        pg = self._pyautogui
        atype = action.type

        if atype == ActionType.CLICK:
            pg.click(action.x, action.y, _pause=False)
        elif atype == ActionType.DOUBLE_CLICK:
            pg.doubleClick(action.x, action.y, _pause=False)
        elif atype == ActionType.RIGHT_CLICK:
            pg.rightClick(action.x, action.y, _pause=False)
        elif atype == ActionType.DRAG:
            pg.moveTo(action.x, action.y, _pause=False)
            pg.dragTo(action.x2, action.y2, duration=0.2, button="left", _pause=False)
        elif atype == ActionType.TYPE:
            # interval=0.015 mimics realistic typing so sites with debouncing
            # catch the keystrokes. Pure paste loses focus-sensitive inputs.
            pg.typewrite(action.text, interval=0.015, _pause=False)
        elif atype == ActionType.KEY:
            pg.press(action.key, _pause=False)
        elif atype == ActionType.HOTKEY:
            # action.key is "ctrl+c", "cmd+shift+3", etc.
            keys = [k.strip() for k in action.key.split("+")]
            pg.hotkey(*keys, _pause=False)
        elif atype == ActionType.SCROLL:
            amount = action.amount or 300
            # pyautogui scroll: positive = up, negative = down. Reverse so our
            # "down" semantically matches V1 (scroll down the page).
            click_units = amount // 20  # pyautogui uses "clicks", ~20px each
            if action.direction == "down":
                pg.scroll(-click_units, _pause=False)
            elif action.direction == "up":
                pg.scroll(click_units, _pause=False)
            elif action.direction == "left":
                pg.hscroll(-click_units, _pause=False)
            elif action.direction == "right":
                pg.hscroll(click_units, _pause=False)
        elif atype == ActionType.WAIT:
            await asyncio.sleep((action.duration_ms or 1000) / 1000)
        elif atype == ActionType.DONE:
            pass  # loop will exit on its own
        else:
            raise ValueError(f"Unsupported action type for OSNativePlatform: {atype}")

    async def teardown(self) -> None:
        if self._mss is not None:
            self._mss.close()
            self._mss = None
