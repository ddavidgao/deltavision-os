"""
OSWorld platform stub. Concrete implementation lands when we wire up the
OSWorld eval harness (separate PR, needs Docker + Ubuntu VM image).

OSWorld spec: https://os-world.github.io/
Actions are keyboard + mouse events dispatched into a virtualized display.
Observations come back as RGB frames.
"""

from typing import Optional

from PIL import Image

from capture.base import Platform


class OSWorldPlatform(Platform):
    """OSWorld VM env wrapper. NOT YET IMPLEMENTED.

    Once wired up, the pattern is:
        env = OSWorldEnv(task_config=...)
        obs = env.reset()         → initial screenshot
        env.step(action)          → new screenshot + reward + done
    """

    def __init__(self, task_id: str, env=None):
        self.task_id = task_id
        self._env = env  # external OSWorld Desktop env instance
        self._frame: Optional[Image.Image] = None

    async def setup(self) -> None:
        if self._env is None:
            raise NotImplementedError(
                "OSWorldPlatform needs an OSWorld env instance. "
                "Install OSWorld and pass env= to the constructor. "
                "This is a scaffold until the eval harness is wired up."
            )
        obs = self._env.reset(task_id=self.task_id)
        self._frame = _frame_from_osworld_obs(obs)

    async def capture(self) -> Image.Image:
        if self._frame is None:
            raise RuntimeError("Platform not set up — call setup() first")
        return self._frame

    async def get_url(self) -> Optional[str]:
        # OSWorld tasks span apps. Some have a URL (browser tasks), most don't.
        # Returning None is safe — classifier falls through to pixel layers.
        return None

    async def execute(self, action) -> None:
        if self._env is None:
            raise RuntimeError("Platform not set up — call setup() first")
        osworld_action = _action_to_osworld(action)
        obs, reward, done, info = self._env.step(osworld_action)
        self._frame = _frame_from_osworld_obs(obs)

    async def teardown(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()


def _frame_from_osworld_obs(obs) -> Image.Image:
    """OSWorld obs format TBD on actual integration — stubbed."""
    import numpy as np
    if isinstance(obs, dict) and "screenshot" in obs:
        arr = obs["screenshot"]
    elif hasattr(obs, "shape"):  # raw ndarray
        arr = obs
    else:
        raise TypeError(f"Don't know how to extract screenshot from OSWorld obs: {type(obs)}")
    return Image.fromarray(np.asarray(arr))


def _action_to_osworld(action):
    """Translate our Action to OSWorld's action format. TBD."""
    raise NotImplementedError(
        "_action_to_osworld pending OSWorld harness wiring. "
        "OSWorld uses pyautogui-style actions internally per their docs; "
        "we may be able to pass through directly."
    )
