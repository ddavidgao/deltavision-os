"""
Platform abstraction. Every observation source (OS-native, OSWorld, VNC)
implements this interface. The agent loop never touches platform-specific code.
"""

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class Platform(ABC):
    """Abstract platform. Owns capture + execute + lifecycle."""

    @abstractmethod
    async def setup(self) -> None:
        """Initialize the platform. Called once before the loop starts.

        For OS: may no-op or check permissions.
        For OSWorld: reset the VM to task initial state.
        """
        ...

    @abstractmethod
    async def capture(self) -> Image.Image:
        """Return the current screen state as a PIL Image.

        Must be cheap (<100ms ideally). Called every loop step.
        """
        ...

    @abstractmethod
    async def get_url(self) -> Optional[str]:
        """Return current URL, or None if this platform has no notion of URL.

        OS-native and OSWorld both return None. The classifier's Layer 1
        (URL change) is effectively disabled when URL is None — transition
        detection falls through to Layers 2-4 (diff ratio, pHash, anchor).
        """
        ...

    @abstractmethod
    async def execute(self, action) -> None:
        """Execute an Action in the platform.

        Action types supported depend on the platform. OS-native supports
        all V2 actions (click, type, scroll, key, drag, double_click,
        right_click, hotkey, wait). OSWorld maps onto its own action space.
        """
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Release resources. Called on shutdown (normal or exceptional)."""
        ...

    # Async context manager helpers so callers can do:
    #   async with platform:
    #       state = await run_agent(...)
    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.teardown()
