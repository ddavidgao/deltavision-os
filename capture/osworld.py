"""
OSWorld platform wrapper.

OSWorld (https://os-world.github.io/) runs tasks inside a virtualized Ubuntu
or Windows desktop, exposing a Flask server in the VM on port 5000. The host-
side `DesktopEnv` from the `desktop_env` package talks to that server over
HTTP and also orchestrates the VM lifecycle (start/stop/snapshot via Docker
or VMware).

API reference (commit c7e54d2 of github.com/xlang-ai/OSWorld):

    env = DesktopEnv(
        provider_name="docker",          # or "vmware"
        os_type="Ubuntu",
        action_space="pyautogui",        # our Actions map cleanly here
        require_a11y_tree=False,         # keep it pixel-only for V2
    )

    obs = env.reset(task_config={...})   # full task dict, NOT just an id
    # obs = {
    #   "screenshot": bytes,             # PNG bytes (NOT ndarray)
    #   "accessibility_tree": str | None,
    #   "terminal": str | None,
    #   "instruction": str,
    # }

    obs, reward, done, info = env.step(action_str, pause=2)
    # action_str is a pyautogui code STRING ('pyautogui.click(100, 200)')
    # or the literal strings 'WAIT', 'FAIL', 'DONE'
    # reward is 0 at step time; scoring happens separately via env.evaluate()

    score = env.evaluate()               # runs the task's declared metric

This file wraps that surface behind our Platform ABC so the agent loop stays
unchanged. Task instantiation and `env.evaluate()` live in the runner
(benchmarks/run_osworld.py) because the Platform ABC is per-episode.
"""

from __future__ import annotations

import io
from typing import Optional

from PIL import Image

from agent.actions import Action, ActionType
from capture.base import Platform


class OSWorldPlatform(Platform):
    """Wraps a pre-instantiated OSWorld `DesktopEnv` object.

    Construction: the runner creates the env, calls `env.reset(task_config=...)`,
    then hands the env + initial obs to this platform. We keep `setup()` cheap
    (no env construction) because the VM is expensive and the runner may
    sequence many tasks against one env.
    """

    def __init__(self, env, initial_obs: Optional[dict] = None):
        """
        env:          DesktopEnv instance from the `desktop_env` package.
                      Already reset — caller owns lifecycle. For a11y-hybrid
                      observations, construct it with `require_a11y_tree=True`.
        initial_obs:  The dict returned by env.reset(). If None, capture()
                      fetches fresh on first call (cheaper for the caller,
                      more HTTP for us; prefer passing it in).
        """
        self._env = env
        self._frame: Optional[Image.Image] = None
        self._instruction: Optional[str] = None
        self._a11y_xml: Optional[str] = None
        if initial_obs is not None:
            self._absorb(initial_obs)

    async def setup(self) -> None:
        # Runner has already reset the env. Nothing to do here unless the
        # caller constructed us without an initial obs.
        if self._frame is None:
            self._frame = await self._fresh_capture()

    async def capture(self) -> Image.Image:
        if self._frame is None:
            self._frame = await self._fresh_capture()
        return self._frame

    async def get_url(self) -> Optional[str]:
        # OSWorld tasks span apps; URL is usually meaningless. Returning None
        # tells the classifier to fall through to diff/pHash/anchor layers.
        return None

    async def get_a11y_xml(self) -> Optional[str]:
        """Return the cached a11y tree from the last env step, if available.

        Non-None only when `DesktopEnv(..., require_a11y_tree=True)` — we
        absorb `obs["accessibility_tree"]` on every reset/step and serve the
        cached copy here. No extra HTTP round-trip per call.
        """
        return self._a11y_xml

    async def execute(self, action: Action) -> None:
        action_str = _action_to_pyautogui_string(action)
        obs, _reward, _done, _info = self._env.step(action_str)
        self._absorb(obs)

    async def teardown(self) -> None:
        # Env lifecycle belongs to the runner; don't close here or we nuke
        # it between tasks.
        pass

    # -- extras the runner needs --

    @property
    def instruction(self) -> Optional[str]:
        """The task's natural-language instruction from env.reset()."""
        return self._instruction

    def evaluate(self) -> float:
        """Run the task's scoring function. Returns 0/1 or a graded score
        per the task's declared metric."""
        return self._env.evaluate()

    # -- private --

    def _absorb(self, obs: dict) -> None:
        """Stash frame + instruction + a11y from an OSWorld obs dict."""
        if "screenshot" in obs:
            self._frame = _pil_from_png_bytes(obs["screenshot"])
        if obs.get("instruction"):
            self._instruction = obs["instruction"]
        # accessibility_tree is an XML string. None when DesktopEnv was built
        # with require_a11y_tree=False; '' when the app didn't expose one.
        self._a11y_xml = obs.get("accessibility_tree")

    async def _fresh_capture(self) -> Image.Image:
        """Ask OSWorld for a new screenshot without taking an action.

        `env._get_obs()` is the internal method; fall back to a no-op WAIT
        step if that's unavailable.
        """
        if hasattr(self._env, "_get_obs"):
            obs = self._env._get_obs()
        else:
            obs, _, _, _ = self._env.step("WAIT")
        return _pil_from_png_bytes(obs["screenshot"])


# -------------------- helpers --------------------


def _pil_from_png_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _q(text: Optional[str]) -> str:
    """Safely embed a string literal into a pyautogui call."""
    if text is None:
        return "''"
    # Use repr to get proper escaping for quotes, backslashes, newlines.
    return repr(text)


def _action_to_pyautogui_string(action: Action) -> str:
    """Translate our typed Action to an OSWorld pyautogui code string.

    OSWorld's pyautogui action_space expects executable Python that calls
    `pyautogui.<method>(...)`. It also accepts 'WAIT', 'FAIL', 'DONE' as
    terminal/no-op sentinels.
    """
    t = action.type
    if t == ActionType.CLICK:
        return f"pyautogui.click({action.x}, {action.y})"
    if t == ActionType.DOUBLE_CLICK:
        return f"pyautogui.doubleClick({action.x}, {action.y})"
    if t == ActionType.RIGHT_CLICK:
        return f"pyautogui.rightClick({action.x}, {action.y})"
    if t == ActionType.DRAG:
        # Two-call sequence — OSWorld accepts multi-line strings.
        return (
            f"pyautogui.moveTo({action.x}, {action.y})\n"
            f"pyautogui.dragTo({action.x2}, {action.y2}, duration=0.2, button='left')"
        )
    if t == ActionType.TYPE:
        return f"pyautogui.typewrite({_q(action.text)}, interval=0.015)"
    if t == ActionType.KEY:
        return f"pyautogui.press({_q(action.key)})"
    if t == ActionType.HOTKEY:
        keys = [k.strip() for k in (action.key or "").split("+")]
        args = ", ".join(_q(k) for k in keys)
        return f"pyautogui.hotkey({args})"
    if t == ActionType.SCROLL:
        amount = action.amount or 300
        clicks = amount // 20  # pyautogui uses scroll "clicks" ~20px
        # Default to 'down' if direction is missing — models like UI-TARS
        # sometimes emit SCROLL without an explicit direction.
        direction = (action.direction or "down").lower()
        if direction == "up":
            return f"pyautogui.scroll({clicks})"
        if direction == "down":
            return f"pyautogui.scroll({-clicks})"
        if direction == "left":
            return f"pyautogui.hscroll({-clicks})"
        if direction == "right":
            return f"pyautogui.hscroll({clicks})"
        # Unknown direction — fall back to down rather than crash the whole task.
        return f"pyautogui.scroll({-clicks})"
    if t == ActionType.WAIT:
        # OSWorld understands "WAIT" as a no-op; fall back to a pyautogui
        # sleep for compatibility since our WAIT carries a duration.
        if action.duration_ms:
            return f"import time; time.sleep({action.duration_ms / 1000:.3f})"
        return "WAIT"
    if t == ActionType.DONE:
        return "DONE"
    raise ValueError(f"Unsupported action for OSWorldPlatform: {t}")
