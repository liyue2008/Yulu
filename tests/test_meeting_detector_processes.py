# pyright: reportMissingImports=false

from pathlib import Path
from unittest.mock import patch

import meeting_detector

ROOT = Path(__file__).resolve().parents[1]


def test_meeting_detector_uses_running_meeting_app_when_titles_are_unavailable() -> None:
    config = dict(meeting_detector.DEFAULT_CONFIG)
    with (
        patch.object(meeting_detector, "collect_windows", return_value=[]),
        patch.object(meeting_detector, "collect_visible_apps", return_value=["Zoom Workplace"]),
    ):
        result = meeting_detector.detect_meeting(config)

    assert result["active"] is True
    assert result["app"] == "Zoom Workplace"
    assert result["fallback"] == "running_process"


def test_meeting_detector_does_not_request_accessibility_when_no_meeting_app_runs() -> None:
    config = dict(meeting_detector.DEFAULT_CONFIG)
    with (
        patch.object(meeting_detector, "collect_windows", return_value=None),
        patch.object(meeting_detector, "collect_visible_apps", return_value=["Finder", "Safari"]),
    ):
        result = meeting_detector.detect_meeting(config)

    assert result["active"] is False
    assert "permission_hint" not in result

    scanner = (ROOT / "yulu/scripts/window_scanner.swift").read_text(encoding="utf-8")
    assert "AXIsProcessTrustedWithOptions" not in scanner
    assert "AXIsProcessTrusted()" in scanner
