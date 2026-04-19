"""
Smoke tests for OSNativePlatform. Capture-only — no action execution.

pyautogui on macOS requires Accessibility permission. We don't exercise
execute() in tests; the platform is capture-only here. Real action exec
is covered by manual integration tests in benchmarks/ (future).
"""

import asyncio
import pytest

from PIL import Image

from deltavision_os.capture.os_native import OSNativePlatform


@pytest.fixture
def platform():
    """Build + setup + teardown an OS-native platform for each test."""
    p = OSNativePlatform()
    asyncio.run(p.setup())
    yield p
    asyncio.run(p.teardown())


class TestCapture:
    def test_returns_pil_image(self, platform):
        img = asyncio.run(platform.capture())
        assert isinstance(img, Image.Image)

    def test_capture_has_sane_dimensions(self, platform):
        img = asyncio.run(platform.capture())
        # Any real monitor is at least 640x480. Retina can be 2880x1800+.
        assert img.width >= 640
        assert img.height >= 480
        assert img.width <= 8000   # 8K safety bound
        assert img.height <= 5000

    def test_capture_is_rgb(self, platform):
        img = asyncio.run(platform.capture())
        assert img.mode == "RGB"

    def test_two_captures_mostly_identical(self, platform):
        """On a quiet desktop, two back-to-back captures should be very similar.
        This catches regressions where capture returns stale / empty data."""
        img1 = asyncio.run(platform.capture())
        import time
        time.sleep(0.05)
        img2 = asyncio.run(platform.capture())

        # Convert to numpy and compute pixel difference ratio
        import numpy as np
        arr1 = np.asarray(img1)
        arr2 = np.asarray(img2)
        if arr1.shape != arr2.shape:
            pytest.fail(f"Capture size changed between calls: {arr1.shape} vs {arr2.shape}")

        diff_mask = np.any(np.abs(arr1.astype(int) - arr2.astype(int)) > 15, axis=-1)
        diff_ratio = diff_mask.mean()
        # Allow some change (clock, cursor) but most pixels must match
        assert diff_ratio < 0.10, f"Two captures 50ms apart differ in {diff_ratio:.1%} of pixels"


class TestURL:
    def test_get_url_returns_none(self, platform):
        """OS-native has no concept of URL. Classifier falls through to
        pixel-based layers when URL is None."""
        url = asyncio.run(platform.get_url())
        assert url is None


class TestLifecycle:
    def test_capture_before_setup_raises(self):
        p = OSNativePlatform()
        with pytest.raises(RuntimeError, match="not set up"):
            asyncio.run(p.capture())

    def test_teardown_clears_mss(self):
        p = OSNativePlatform()
        asyncio.run(p.setup())
        assert p._mss is not None
        asyncio.run(p.teardown())
        assert p._mss is None
