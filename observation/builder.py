"""
Builds typed Observation objects from raw pipeline outputs.
"""

from typing import Optional, List
from PIL import Image

from agent.actions import Action
from vision.diff import DiffResult
from .types import FullFrameObservation, DeltaObservation, Observation
from .a11y import A11yObservation, build_a11y_observation


def build_observation(
    obs_type: str,
    task: str,
    step: int,
    last_action: Optional[Action],
    # Full frame args
    frame: Optional[Image.Image] = None,
    url: str = "",
    trigger_reason: str = "",
    # Delta args
    diff_result: Optional[DiffResult] = None,
    crops: Optional[List[dict]] = None,
    action_had_effect: bool = False,
    no_change_count: int = 0,
    text_deltas: Optional[List[dict]] = None,
    current_frame: Optional[Image.Image] = None,
    # A11y-hybrid args (V2 unlock, matches V1's v1.0.2 DOM+focus pattern).
    # `a11y_xml` is the raw XML string from Platform.get_a11y_xml(). If
    # None, the caller opted out or the platform doesn't support it — the
    # builder returns an `A11yObservation(status='disabled')` marker.
    a11y_xml: Optional[str] = None,
) -> Observation:
    """
    Factory for building the right observation type.
    The agent loop calls this — model backends consume the result.
    """
    a11y = _build_a11y(obs_type, a11y_xml, diff_result)

    if obs_type == "full_frame":
        return FullFrameObservation(
            obs_type="full_frame",
            task=task,
            step=step,
            last_action=last_action,
            frame=frame,
            url=url,
            trigger_reason=trigger_reason,
            a11y=a11y,
        )

    return DeltaObservation(
        obs_type="delta",
        task=task,
        step=step,
        last_action=last_action,
        diff_result=diff_result,
        crops=crops or [],
        action_had_effect=action_had_effect,
        no_change_count=no_change_count,
        text_deltas=text_deltas or [],
        current_frame=current_frame,
        a11y=a11y,
    )


def _build_a11y(
    obs_type: str,
    a11y_xml: Optional[str],
    diff_result: Optional[DiffResult],
) -> Optional[A11yObservation]:
    """Produce the A11yObservation from raw XML.

    On the delta path we prune to nodes intersecting the changed bboxes;
    on the full_frame path we send the top-N nodes by order of discovery.
    Passing xml=None (platform doesn't provide it) returns None instead of
    an A11yObservation — that's the "platform has no a11y" signal.
    """
    if a11y_xml is None:
        return None  # not an A11yObservation with status='disabled';
                     # `None` means the platform doesn't even have the
                     # concept. Distinct from 'disabled' (caller opted out).
    changed_bboxes = None
    if obs_type == "delta" and diff_result is not None:
        changed_bboxes = list(diff_result.changed_bboxes)
    return build_a11y_observation(a11y_xml, changed_bboxes=changed_bboxes)
