"""
Naive-user install invariants.

These tests codify the lesson from V1's v1.0.2 packaging bug (import broken
from fresh venv) and the V2 pre-nesting shadowing bug (local `vision/` or
`agent/` dir silently shadowed installed modules). Write tests the way a
new user would use the package — not the way an insider imports from the
source tree.

If these fail, a pip-installed user in a fresh venv won't be able to use
the library. CI should run these on every PR.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_in(cwd: str, code: str) -> subprocess.CompletedProcess:
    """Spawn a Python subprocess in a specific cwd, running the given code.

    We use subprocess instead of in-process exec so that Python's module
    resolution actually treats the cwd as its starting sys.path[0] — the
    main way a naive user's local dirs can shadow an install.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )


class TestNaiveImport:
    """The README-documented import path must work from anywhere."""

    def test_import_from_tmp(self, tmp_path):
        """Fresh cwd with no relation to the source tree — the core case."""
        r = _run_in(str(tmp_path), "import deltavision_os; "
                                   "print(deltavision_os.__version__)")
        assert r.returncode == 0, f"stderr:\n{r.stderr}"
        assert r.stdout.strip(), "no version printed"

    def test_public_api_documented_symbols(self):
        """Everything the README names as a class/function must be
        importable directly from the top-level package."""
        import deltavision_os as dv
        promised = [
            # platforms
            "Platform", "OSNativePlatform", "OSWorldPlatform",
            # agent loop + state
            "run_agent", "AgentState", "Action", "ActionType",
            # config + safety
            "DeltaVisionConfig", "SafetyLayer",
            # observations
            "Observation", "FullFrameObservation", "DeltaObservation",
            "build_observation",
            # a11y hybrid
            "A11yObservation", "A11yNode",
            "build_a11y_observation", "parse_a11y_xml",
        ]
        missing = [n for n in promised if not hasattr(dv, n)]
        assert not missing, (
            f"deltavision_os missing public symbols: {missing}. "
            f"A naive user's first attempt will fail."
        )

    def test_version_populated(self):
        import deltavision_os
        assert hasattr(deltavision_os, "__version__")
        assert deltavision_os.__version__  # non-empty


class TestShadowResistance:
    """A user's cwd having a `vision/` or `agent/` dir must not break us.

    Regression guard for the V2 pre-nesting bug: before `deltavision_os/`
    was a proper nested package, any user directory with one of our
    generic package names would shadow internal imports and crash the
    library with confusing `ImportError: cannot import name X from vision`
    messages. The fix was to nest everything under `deltavision_os/`; this
    test confirms the fix holds.
    """

    @pytest.mark.parametrize("shadow_dir", [
        "vision", "agent", "capture", "model", "observation", "results",
    ])
    def test_import_survives_single_shadowed_dir(self, tmp_path, shadow_dir):
        (tmp_path / shadow_dir).mkdir()
        (tmp_path / shadow_dir / "__init__.py").write_text(
            "# Shadow: a user's unrelated project may have this name.\n"
            "raise RuntimeError('shadow wins — deltavision_os install is broken')"
        )
        r = _run_in(str(tmp_path),
                    "import deltavision_os; "
                    "from deltavision_os import OSNativePlatform, run_agent; "
                    "print('ok')")
        assert r.returncode == 0, (
            f"Shadow test failed with dir={shadow_dir!r}. "
            f"A local {shadow_dir}/ dir is shadowing our install.\n"
            f"stderr:\n{r.stderr}"
        )
        assert "ok" in r.stdout

    def test_import_survives_all_shadowed_dirs_at_once(self, tmp_path):
        """Worst case: user's project has ALL our internal package names."""
        for d in ("vision", "agent", "capture", "model", "observation",
                  "results"):
            (tmp_path / d).mkdir()
            (tmp_path / d / "__init__.py").write_text(
                "raise RuntimeError('shadow wins')"
            )
        (tmp_path / "config.py").write_text("raise RuntimeError('shadow wins')")
        (tmp_path / "safety.py").write_text("raise RuntimeError('shadow wins')")

        r = _run_in(str(tmp_path),
                    "import deltavision_os; "
                    "from deltavision_os import OSNativePlatform, run_agent, "
                    "DeltaVisionConfig, SafetyLayer, build_a11y_observation; "
                    "print('ok')")
        assert r.returncode == 0, (
            f"Full-shadow test failed. This means deltavision_os is NOT a "
            f"properly nested package — user projects with generic subdir "
            f"names will break the install.\nstderr:\n{r.stderr}"
        )
        assert "ok" in r.stdout


class TestSubpackageImports:
    """Documented sub-imports must work too, for users who want finer
    access (e.g. `from deltavision_os.agent.loop import run_agent`)."""

    def test_nested_imports(self, tmp_path):
        code = textwrap.dedent("""
            from deltavision_os.agent.loop import run_agent
            from deltavision_os.capture.os_native import OSNativePlatform
            from deltavision_os.observation.a11y import build_a11y_observation
            from deltavision_os.config import DeltaVisionConfig
            print('ok')
        """)
        r = _run_in(str(tmp_path), code)
        assert r.returncode == 0, f"stderr:\n{r.stderr}"
        assert "ok" in r.stdout
