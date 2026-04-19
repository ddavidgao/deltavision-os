"""
Observation dataclasses. These are what the model receives as input.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image

from deltavision_os.agent.actions import Action
from deltavision_os.vision.diff import DiffResult
from deltavision_os.observation.a11y import A11yObservation


@dataclass
class Observation:
    """Base observation — common fields."""

    obs_type: str  # "full_frame" or "delta"
    task: str
    step: int
    last_action: Optional[Action]


@dataclass
class FullFrameObservation(Observation):
    """Sent on initial load, navigation, or forced refresh."""

    frame: Optional[Image.Image] = None
    url: str = ""
    trigger_reason: str = ""  # "initial" | "url_change" | "diff_ratio" | "phash" | "anchor_loss" | "force_refresh_no_effect"
    # Optional a11y payload. None = platform doesn't support it; an
    # A11yObservation object (with status='disabled', 'ok', 'timeout', etc.)
    # means we attempted to fetch. Model backends serialize into the prompt.
    a11y: Optional[A11yObservation] = None

    def __post_init__(self):
        self.obs_type = "full_frame"


@dataclass
class DeltaObservation(Observation):
    """Sent when the page is the same but something changed (or didn't)."""

    diff_result: Optional[DiffResult] = None
    crops: List[dict] = field(default_factory=list)
    action_had_effect: bool = False
    no_change_count: int = 0
    # Level 1 optimization: text deltas extracted via OCR
    text_deltas: List[dict] = field(default_factory=list)  # [{"bbox": ..., "before": str, "after": str}]
    # Current frame for backends that want a single annotated screenshot
    current_frame: Optional[Image.Image] = None
    # A11y payload pruned to changed regions + focused element. See
    # observation/a11y.py for the schema and observation/builder.py for how
    # it's populated from capture/*.get_a11y_xml().
    a11y: Optional[A11yObservation] = None

    def __post_init__(self):
        self.obs_type = "delta"
