"""
Unit tests for observation/a11y.py — parser, filter, changed-region pruner.

No live OSWorld needed; uses hand-crafted XML matching the namespaced
attribute shape that OSWorld's `desktop_env/server/main.py::_create_atspi_node`
emits. See observation/a11y.py for the namespace constants.
"""

from __future__ import annotations

import textwrap

import pytest

from observation.a11y import (
    A11yNode,
    A11yObservation,
    build_a11y_observation,
    parse_a11y_xml,
    _intersect_ratio,
)


# Minimal but representative XML — mirrors the GIMP-menu scenario from the
# Phase 3 diagnostic. One focused spinbox, two visible menu items, one
# hidden-showing=false button that should be filtered out, one interactive
# link.
GIMP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <desktop-frame
        xmlns:st="https://accessibility.ubuntu.example.org/ns/state"
        xmlns:cp="https://accessibility.ubuntu.example.org/ns/component"
        xmlns:val="https://accessibility.ubuntu.example.org/ns/value">
      <application name="GIMP">
        <frame name="Untitled-1 - GIMP">
          <menu-bar>
            <menu name="File"
                  st:visible="true" st:showing="true" st:enabled="true"
                  cp:screencoord="(52, 78)" cp:size="(40, 30)"/>
            <menu name="Colors"
                  st:visible="true" st:showing="true" st:enabled="true"
                  cp:screencoord="(318, 78)" cp:size="(50, 30)"/>
            <menu name="OffscreenItem"
                  st:visible="true" st:showing="false" st:enabled="true"
                  cp:screencoord="(900, 78)" cp:size="(40, 30)"/>
          </menu-bar>
          <dialog name="Hue-Saturation">
            <spin-button name="Saturation"
                         st:visible="true" st:showing="true"
                         st:enabled="true" st:focused="true"
                         cp:screencoord="(550, 510)" cp:size="(100, 30)"
                         val:value="1.0"/>
            <push-button name="OK"
                         st:visible="true" st:showing="true" st:enabled="true"
                         cp:screencoord="(700, 600)" cp:size="(60, 30)"/>
          </dialog>
          <link name="Helpdoc"
                st:visible="true" st:showing="true" st:enabled="true"
                cp:screencoord="(1000, 20)" cp:size="(80, 20)"/>
        </frame>
      </application>
    </desktop-frame>
""")


class TestParseA11yXml:
    def test_parses_visible_interactive_nodes(self):
        nodes, raw_count, truncated = parse_a11y_xml(GIMP_FIXTURE)
        roles = [n.role for n in nodes]
        assert "menu" in roles
        assert "spin-button" in roles
        assert "push-button" in roles
        assert "link" in roles
        assert not truncated
        assert raw_count >= len(nodes)

    def test_filters_showing_false(self):
        nodes, _, _ = parse_a11y_xml(GIMP_FIXTURE)
        names = {n.name for n in nodes}
        assert "OffscreenItem" not in names

    def test_filters_root_frames(self):
        nodes, _, _ = parse_a11y_xml(GIMP_FIXTURE)
        roles = {n.role for n in nodes}
        assert "desktop-frame" not in roles
        # application / frame have no bboxes in the fixture → filtered out
        # regardless of role-whitelist state.
        assert "application" not in roles

    def test_extracts_bbox_and_value(self):
        nodes, _, _ = parse_a11y_xml(GIMP_FIXTURE)
        sat = next(n for n in nodes if n.name == "Saturation")
        assert sat.bbox == (550, 510, 100, 30)
        assert sat.value == "1.0"
        assert sat.focused is True

    def test_unique_ids(self):
        nodes, _, _ = parse_a11y_xml(GIMP_FIXTURE)
        ids = [n.id for n in nodes]
        assert len(ids) == len(set(ids))

    def test_parse_error_returns_empty(self):
        nodes, raw_count, truncated = parse_a11y_xml("<not-well-formed")
        assert nodes == []
        assert raw_count == 0
        assert not truncated

    def test_truncation(self):
        # Build a synthetic fixture with more than 5 interactive nodes,
        # cap at 3.
        many = "<root " + _xmlns() + ">" + "".join(
            f'<push-button name="b{i}" st:visible="true" st:showing="true" '
            f'st:enabled="true" cp:screencoord="({i * 10}, 0)" '
            f'cp:size="(8, 8)"/>' for i in range(10)
        ) + "</root>"
        nodes, _, truncated = parse_a11y_xml(many, max_nodes=3)
        assert len(nodes) == 3
        assert truncated


def _xmlns() -> str:
    return (
        'xmlns:st="https://accessibility.ubuntu.example.org/ns/state" '
        'xmlns:cp="https://accessibility.ubuntu.example.org/ns/component"'
    )


class TestIntersectRatio:
    def test_full_overlap(self):
        assert _intersect_ratio((10, 10, 100, 100), [(0, 0, 200, 200)]) == 1.0

    def test_no_overlap(self):
        assert _intersect_ratio((10, 10, 100, 100), [(500, 500, 10, 10)]) == 0.0

    def test_partial_overlap(self):
        # 50x50 overlap on a 100x100 node = 0.25
        ratio = _intersect_ratio((0, 0, 100, 100), [(50, 50, 100, 100)])
        assert ratio == pytest.approx(0.25)

    def test_zero_area_node(self):
        assert _intersect_ratio((0, 0, 0, 0), [(0, 0, 50, 50)]) == 0.0


class TestBuildA11yObservation:
    def test_disabled_returns_disabled_status(self):
        obs = build_a11y_observation(None)
        assert obs.status == "disabled"
        assert obs.changed_elements == []

    def test_empty_xml_returns_unavailable(self):
        obs = build_a11y_observation("")
        assert obs.status == "unavailable"

    def test_full_frame_path_no_bboxes(self):
        """When changed_bboxes is None, send everything up to max_elements."""
        obs = build_a11y_observation(GIMP_FIXTURE, changed_bboxes=None,
                                     max_elements=10)
        assert obs.status in ("ok", "truncated")
        assert len(obs.changed_elements) >= 4
        assert obs.focused_element is not None
        assert obs.focused_element.name == "Saturation"

    def test_delta_path_prunes_to_intersecting(self):
        """Only nodes inside the changed bbox (+ focused) should be kept."""
        # Changed region covers the Colors menu (318, 78, 50, 30).
        obs = build_a11y_observation(
            GIMP_FIXTURE, changed_bboxes=[(310, 70, 70, 50)])
        names = {n.name for n in obs.changed_elements}
        assert "Colors" in names
        # Focused element ALWAYS included, even if outside changed region.
        assert "Saturation" in names
        # Items far from the changed bbox shouldn't be here.
        assert "Helpdoc" not in names

    def test_delta_path_ranks_by_intersect(self):
        # Changed region fully covers File + partially Colors.
        obs = build_a11y_observation(
            GIMP_FIXTURE, changed_bboxes=[(50, 75, 50, 35)])
        # File is fully inside the changed bbox → highest ratio.
        top = obs.changed_elements[0]
        assert top.name in ("File", "Saturation")  # Saturation is focused-prepended

    def test_prompt_text_compact(self):
        obs = build_a11y_observation(
            GIMP_FIXTURE, changed_bboxes=[(310, 70, 70, 50)])
        text = obs.prompt_text(max_elements=5)
        assert "FOCUS" in text
        assert "Saturation" in text
        assert "Colors" in text
        # Compactness sanity: should be short enough for a model prompt.
        assert text.count("\n") < 20

    def test_too_large_fallback(self):
        # Explicitly feed a synthetically huge string (no actual parse).
        huge = " " * (3_000_000)
        obs = build_a11y_observation(huge, changed_bboxes=[])
        assert obs.status == "too_large"
        assert obs.raw_bytes == len(huge)
