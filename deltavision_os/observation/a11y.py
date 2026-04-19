"""
Accessibility-tree hybrid observations for DeltaVision-OS.

When the classifier sends a DELTA observation (nothing big changed visually),
the model can struggle to recover from coordinate misses — it only sees small
crops, not "where am I on the window." Mirrors V1's v1.0.2 DOM+focus unlock:
we augment pixel observations with structured text about UI elements.

Design: use the pixel diff bboxes as the GATE for which a11y nodes enter the
prompt. Novelty vs prior art (UFO2 filters by interactive-visibility, OSWorld
filters by role whitelist): DeltaVision-OS filters by *what changed* + what's
focused.

OSWorld returns a11y as an XML string via `obs["accessibility_tree"]`. Node
tag IS the role. Attributes use namespaces:

  st:focused / st:visible / st:showing / st:enabled   (state)
  cp:screencoord="(x, y)" / cp:size="(w, h)"          (component)
  val:value                                            (value, optional)

Root is <desktop-frame> (Linux), <desktop> (Windows), or similar on macOS.

We parse with xml.etree (stdlib; no new deps). XML is handed to us as text,
not a file path.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, Optional


# XML namespaces OSWorld uses (Ubuntu/Linux primary; Windows/macOS mirror).
NS_STATE = "https://accessibility.ubuntu.example.org/ns/state"
NS_COMPONENT = "https://accessibility.ubuntu.example.org/ns/component"
NS_VALUE = "https://accessibility.ubuntu.example.org/ns/value"
NS_ATTRIBUTES = "https://accessibility.ubuntu.example.org/ns/attributes"


# Same role whitelist OSWorld's own `filter_nodes` uses — proven to keep the
# model's attention on interactive elements without losing the window chrome
# needed for navigation.
INTERACTIVE_ROLES = frozenset([
    "push-button", "button", "toggle-button",
    "menu", "menu-item", "menu-bar", "menubar",
    "check-box", "radio-button",
    "entry", "password-text", "text", "document-text",
    "combo-box", "list", "list-item", "list-box",
    "slider", "spin-button", "spinbox", "scroll-bar",
    "tab", "tab-list", "page-tab", "tab-panel",
    "icon", "image", "canvas",
    "tree-item", "tree-table-cell", "table-cell",
    "link", "label",
    "dialog", "window", "frame", "panel",
    # Windows UIA shows these with underscores sometimes:
    "Button", "Menu", "MenuItem", "Edit", "ComboBox", "CheckBox",
    "RadioButton", "Slider", "TabItem", "Image", "Hyperlink",
])


# Default maximum a11y-tree-fetch latency before we fall back to
# screenshot-only. Calc-with-big-sheets can take 10 minutes (OSWorld #185).
DEFAULT_A11Y_TIMEOUT_S = 5.0

# Maximum bytes of raw XML we'll even try to parse. Beyond this, we fall back
# to "unavailable" rather than pay the parse + token cost.
MAX_XML_BYTES = 2_000_000  # 2 MB


@dataclass
class A11yNode:
    """One element in the parsed tree. Mirrors the schema from our research."""
    id: int
    role: str
    name: str
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    text: str = ""
    value: Optional[str] = None
    focused: bool = False
    enabled: bool = True
    # Fraction of bbox area covered by the changed mask. Set by the pruner;
    # 0.0 when the node is not intersecting any changed region (focused
    # fallback). Useful as a relevance ranking in prompts.
    intersect_ratio: float = 0.0

    def as_dict(self) -> dict:
        d = {
            "id": self.id,
            "role": self.role,
            "name": self.name,
            "bbox": list(self.bbox),
            "focused": self.focused,
            "enabled": self.enabled,
        }
        if self.text:
            d["text"] = self.text
        if self.value is not None:
            d["value"] = self.value
        if self.intersect_ratio > 0:
            d["intersect_ratio"] = round(self.intersect_ratio, 3)
        return d


@dataclass
class A11yObservation:
    """The hybrid a11y payload attached to a DV observation.

    `status` is an honest signal to the model: 'ok' means the tree parsed
    fine, 'truncated' means we capped the node count, 'timeout' means the
    server took too long, 'unavailable' means the tree was empty/missing/
    unparseable, 'disabled' means the caller chose not to fetch.
    """
    status: str  # 'ok' | 'truncated' | 'timeout' | 'unavailable' | 'disabled' | 'too_large'
    changed_elements: list[A11yNode] = field(default_factory=list)
    focused_element: Optional[A11yNode] = None
    raw_node_count: int = 0  # pre-pruning (for telemetry)
    raw_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "changed_elements": [n.as_dict() for n in self.changed_elements],
            "focused_element": self.focused_element.as_dict() if self.focused_element else None,
            "raw_node_count": self.raw_node_count,
            "raw_bytes": self.raw_bytes,
        }

    def prompt_text(self, max_elements: int = 20) -> str:
        """Render as compact text for inclusion in a model prompt.

        Format: one line per element — `id|role|name|bbox|[focused|value|text...]`.
        Keeps the token cost visible to the caller (one line ≈ 10-30 tokens).
        """
        if self.status == "disabled":
            return ""
        if self.status != "ok" and self.status != "truncated":
            return f"[a11y: {self.status}]"

        lines = [f"[a11y: {self.status}, {self.raw_node_count} raw nodes, showing "
                 f"{min(len(self.changed_elements), max_elements)}]"]
        if self.focused_element:
            f = self.focused_element
            focus_line = (f"  FOCUS id={f.id} {f.role} '{f.name}' "
                          f"bbox={list(f.bbox)}")
            if f.value:
                focus_line += f" value={f.value!r}"
            lines.append(focus_line)
        for n in self.changed_elements[:max_elements]:
            bits = [f"id={n.id}", n.role]
            if n.name:
                bits.append(f"'{n.name}'")
            bits.append(f"bbox={list(n.bbox)}")
            if n.focused:
                bits.append("focused")
            if not n.enabled:
                bits.append("disabled")
            if n.value is not None:
                bits.append(f"value={n.value!r}")
            if n.intersect_ratio > 0:
                bits.append(f"covers={n.intersect_ratio:.2f}")
            if n.text:
                t = n.text[:40].replace("\n", " ")
                bits.append(f"text={t!r}")
            lines.append("  " + " ".join(bits))
        if len(self.changed_elements) > max_elements:
            lines.append(f"  ... +{len(self.changed_elements) - max_elements} more")
        return "\n".join(lines)


# -------------------- parsing --------------------


_COORD_RE = re.compile(r"[-+]?\d+")


def _parse_pair(val: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse 'cp:screencoord' / 'cp:size' values like '(x, y)' or '(w, h)'."""
    if not val:
        return None
    nums = _COORD_RE.findall(val)
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


def _attr(elem: ET.Element, ns: str, local: str) -> Optional[str]:
    """Read a namespaced attribute, tolerating both '{ns}local' and 'prefix:local'."""
    return elem.get(f"{{{ns}}}{local}") or elem.get(local)


def _is_truthy(val: Optional[str]) -> bool:
    return val is not None and val.strip().lower() in ("true", "1", "yes")


def _bbox_of(elem: ET.Element) -> Optional[tuple[int, int, int, int]]:
    coord = _parse_pair(_attr(elem, NS_COMPONENT, "screencoord"))
    size = _parse_pair(_attr(elem, NS_COMPONENT, "size"))
    if not coord or not size:
        return None
    x, y = coord
    w, h = size
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _role_of(elem: ET.Element) -> str:
    """Role is the XML tag (stripped of any namespace)."""
    tag = elem.tag
    if tag.startswith("{"):
        tag = tag.split("}", 1)[1]
    return tag


def _visible(elem: ET.Element) -> bool:
    # Per OSWorld's filter_nodes: require visible AND showing.
    vis = _is_truthy(_attr(elem, NS_STATE, "visible"))
    showing = _is_truthy(_attr(elem, NS_STATE, "showing"))
    return vis and showing


def parse_a11y_xml(xml_str: str, max_nodes: int = 2000
                   ) -> tuple[list[A11yNode], int, bool]:
    """Parse OSWorld a11y XML into a flat list of candidate nodes.

    Returns (candidates, raw_count, truncated). Candidates are already
    interactivity-filtered (role whitelist + visible/showing + enabled +
    bbox present + has name/text or is focused).

    `raw_count` is the total number of XML elements scanned pre-filter
    (for telemetry). `truncated` is True if we hit `max_nodes`.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return [], 0, False

    candidates: list[A11yNode] = []
    raw_count = 0
    truncated = False
    next_id = 0

    for elem in root.iter():
        raw_count += 1
        if len(candidates) >= max_nodes:
            truncated = True
            break

        role = _role_of(elem)
        if role in ("desktop-frame", "desktop", "root"):
            continue

        bbox = _bbox_of(elem)
        if bbox is None:
            continue

        if not _visible(elem):
            continue

        name = (elem.get("name") or "").strip()
        text = (elem.text or "").strip()
        value_raw = _attr(elem, NS_VALUE, "value")
        focused = _is_truthy(_attr(elem, NS_STATE, "focused"))
        enabled = _is_truthy(_attr(elem, NS_STATE, "enabled")) or _is_truthy(
            _attr(elem, NS_STATE, "sensitive"))

        interactive = role in INTERACTIVE_ROLES
        if not interactive and not focused:
            # Not interactive and not focused; skip.
            continue

        # Need at least SOME identifying info — a nameless text span is noise.
        if not (name or text or focused or value_raw):
            continue

        candidates.append(A11yNode(
            id=next_id,
            role=role,
            name=name,
            bbox=bbox,
            text=text,
            value=value_raw,
            focused=focused,
            enabled=enabled,
        ))
        next_id += 1

    return candidates, raw_count, truncated


# -------------------- pruning to changed regions --------------------


def _intersect_ratio(node_bbox: tuple[int, int, int, int],
                     changed_bboxes: Iterable[tuple[int, int, int, int]]) -> float:
    """Fraction of node_bbox area covered by the union of any changed bboxes
    that intersect it. Coarse (counts overlaps separately) but cheap and
    monotone — what we need for ranking."""
    nx, ny, nw, nh = node_bbox
    if nw <= 0 or nh <= 0:
        return 0.0
    narea = nw * nh
    covered = 0
    for cx, cy, cw, ch in changed_bboxes:
        ix1 = max(nx, cx)
        iy1 = max(ny, cy)
        ix2 = min(nx + nw, cx + cw)
        iy2 = min(ny + nh, cy + ch)
        if ix2 > ix1 and iy2 > iy1:
            covered += (ix2 - ix1) * (iy2 - iy1)
    return min(covered / narea, 1.0)


def build_a11y_observation(
    xml_str: Optional[str],
    changed_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    max_elements: int = 20,
) -> A11yObservation:
    """Run the full pipeline: parse → filter → prune to changed-region
    intersect + focused fallback → A11yObservation.

    changed_bboxes=None means "send everything" (full_frame style — caller
    doesn't need change-region gating). Pass the DiffResult's changed_bboxes
    on delta paths.
    """
    if xml_str is None:
        return A11yObservation(status="disabled")
    if not xml_str:
        return A11yObservation(status="unavailable", raw_bytes=0)
    n_bytes = len(xml_str)
    if n_bytes > MAX_XML_BYTES:
        return A11yObservation(status="too_large", raw_bytes=n_bytes)

    candidates, raw_count, truncated = parse_a11y_xml(xml_str)
    if not candidates:
        return A11yObservation(status="unavailable", raw_bytes=n_bytes,
                               raw_node_count=raw_count)

    focused = next((n for n in candidates if n.focused), None)

    if changed_bboxes is None:
        changed = candidates[:max_elements]
    else:
        for n in candidates:
            n.intersect_ratio = _intersect_ratio(n.bbox, changed_bboxes)
        # Keep nodes that actually intersect; fall back to ranking by area if
        # nothing intersected (can happen when the diff is all in unlabeled
        # pixel noise).
        changed = [n for n in candidates if n.intersect_ratio > 0.0]
        changed.sort(key=lambda n: n.intersect_ratio, reverse=True)
        changed = changed[:max_elements]
        if focused and focused not in changed:
            # Always include the focused element, even if outside changed
            # regions — it's the cheapest signal and often what the model
            # needs to know next.
            changed = [focused] + changed[: max_elements - 1]

    status = "truncated" if truncated else "ok"
    return A11yObservation(
        status=status,
        changed_elements=changed,
        focused_element=focused,
        raw_node_count=raw_count,
        raw_bytes=n_bytes,
    )
