"""
deltavision_os — shim namespace that re-exports the public API.

The internal code lives as flat top-level packages (`agent/`, `capture/`,
`observation/`, etc.) for historical reasons. This shim lets users do

    from deltavision_os import OSNativePlatform, run_agent, DeltaVisionConfig

without caring about the internal layout. It also reserves the namespace
so a downstream project's local `vision/` or `agent/` directory is less
likely to shadow an install (the more collision-prone flat modules are
still exposed at site-packages root — see the namespace-pollution note
in the README for the proper fix roadmap).

If you're reading the source and wondering why imports inside the package
itself look like `from deltavision_os.capture.os_native import ...` instead of
`from deltavision_os.capture.os_native import ...`, that's why. The shim
re-exports; it doesn't reshape the codebase.
"""

from __future__ import annotations

__version__ = "0.1.0a0"

# Platform ABCs + concrete platforms
from deltavision_os.capture.base import Platform
from deltavision_os.capture.os_native import OSNativePlatform
from deltavision_os.capture.osworld import OSWorldPlatform

# Agent loop + state
from deltavision_os.agent.loop import run_agent
from deltavision_os.agent.state import AgentState
from deltavision_os.agent.actions import Action, ActionType, parse_action

# Config + safety
from deltavision_os.config import DeltaVisionConfig
from deltavision_os.safety import SafetyLayer

# Observation types + builder + a11y
from deltavision_os.observation.builder import build_observation
from deltavision_os.observation.types import Observation, FullFrameObservation, DeltaObservation
from deltavision_os.observation.a11y import (
    A11yObservation,
    A11yNode,
    build_a11y_observation,
    parse_a11y_xml,
)

# Model backends (import lazily if the optional deps are missing)
try:
    from deltavision_os.model.claude import ClaudeModel  # noqa: F401
except ImportError:  # pragma: no cover
    ClaudeModel = None  # type: ignore[assignment]

try:
    from deltavision_os.model.openai import OpenAIModel  # noqa: F401
except ImportError:  # pragma: no cover
    OpenAIModel = None  # type: ignore[assignment]

try:
    from deltavision_os.model.scripted import ScriptedModel  # noqa: F401
except ImportError:  # pragma: no cover
    ScriptedModel = None  # type: ignore[assignment]

try:
    from deltavision_os.model.llamacpp import LlamaCppModel  # noqa: F401
except ImportError:  # pragma: no cover
    LlamaCppModel = None  # type: ignore[assignment]

# Results store
try:
    from deltavision_os.results.store import ResultStore  # noqa: F401
except ImportError:  # pragma: no cover
    ResultStore = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    # platforms
    "Platform",
    "OSNativePlatform",
    "OSWorldPlatform",
    # agent loop
    "run_agent",
    "AgentState",
    "Action",
    "ActionType",
    "parse_action",
    # config + safety
    "DeltaVisionConfig",
    "SafetyLayer",
    # observations
    "Observation",
    "FullFrameObservation",
    "DeltaObservation",
    "build_observation",
    # a11y
    "A11yObservation",
    "A11yNode",
    "build_a11y_observation",
    "parse_a11y_xml",
    # models (may be None if optional deps missing)
    "ClaudeModel",
    "OpenAIModel",
    "ScriptedModel",
    "LlamaCppModel",
    # results
    "ResultStore",
]
