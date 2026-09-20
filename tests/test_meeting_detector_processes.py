# pyright: reportMissingImports=false

from pathlib import Path
from types import SimpleNamespace
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


def test_meeting_detector_uses_active_lark_cli_meeting_without_accessibility() -> None:
    config = {
        **meeting_detector.DEFAULT_CONFIG,
        "lark_cli_active_meeting": True,
    }
    command = "/Users/test/.npm-global/bin/lark-cli"
    payload = {
        "ok": True,
        "data": {
            "meetings": [{
                "meeting_id": "7687635521951452927",
                "meeting_title": "Design review",
            }],
        },
    }
    with (
        patch.object(meeting_detector, "_lark_cli_executable", return_value=command),
        patch.object(
            meeting_detector.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout=meeting_detector.json.dumps(payload),
            ),
        ) as run,
    ):
        result = meeting_detector.detect_meeting(config)

    assert result == {
        "active": True,
        "title": "Design review",
        "app": "Lark",
        "window": "",
        "signature": meeting_detector.signature("Lark", "7687635521951452927"),
        "fallback": "lark_cli",
    }
    assert run.call_args.args[0] == [
        command,
        "vc",
        "+meeting-list-active",
        "--as",
        "user",
        "--format",
        "json",
    ]


def test_meeting_detector_uses_lark_meeting_process_without_accessibility() -> None:
    config = dict(meeting_detector.DEFAULT_CONFIG)
    with (
        patch.object(meeting_detector, "collect_windows", return_value=[]),
        patch.object(
            meeting_detector,
            "collect_running_process_commands",
            return_value=[
                "/Applications/LarkSuite.app/Contents/Frameworks/Lark Helper "
                "--type=renderer --scene=byteview",
            ],
        ),
        patch.object(meeting_detector, "collect_visible_apps", return_value=["Lark"]),
    ):
        result = meeting_detector.detect_meeting(config)

    assert result == {
        "active": True,
        "title": "Lark Meeting",
        "app": "Lark",
        "window": "",
        "signature": meeting_detector.signature("Lark", "meeting-process"),
        "fallback": "meeting_process",
    }


def test_lark_main_process_alone_does_not_look_like_a_meeting() -> None:
    config = dict(meeting_detector.DEFAULT_CONFIG)
    with (
        patch.object(meeting_detector, "collect_windows", return_value=[]),
        patch.object(
            meeting_detector,
            "collect_running_process_commands",
            return_value=["/Applications/LarkSuite.app/Contents/MacOS/Lark"],
        ),
        patch.object(meeting_detector, "collect_visible_apps", return_value=["Lark"]),
    ):
        result = meeting_detector.detect_meeting(config)

    assert result["active"] is False


def test_lark_auto_record_dispatches_start_without_a_prompt() -> None:
    config = {**meeting_detector.DEFAULT_CONFIG, "auto_record_apps": ["Lark"]}
    calls = []
    with patch.object(
        meeting_detector.subprocess,
        "Popen",
        side_effect=lambda arguments, **options: calls.append((arguments, options)),
    ):
        mode = meeting_detector.dispatch_recording(
            config,
            {
                "app": "Lark",
                "signature": meeting_detector.signature("Lark", "meeting-process"),
            },
            "检测到会议：Lark Meeting",
        )

    assert mode == "automatic"
    assert calls[0][0] == [
        meeting_detector.sys.executable,
        str(meeting_detector.SCRIPT_DIR / "meeting_daemon.py"),
        "start",
        "检测到会议：Lark Meeting",
        f"detected::{meeting_detector.signature('Lark', 'meeting-process')}",
    ]


def test_auto_recording_stops_after_the_same_meeting_disappears() -> None:
    meeting_id = "detected::lark-signature"
    state = {
        "prompted": {},
        "auto_recording": {
            "meeting_id": meeting_id,
            "signature": "lark-signature",
            "started_at": 50.0,
        },
    }
    saved = []
    config = {**meeting_detector.DEFAULT_CONFIG, "auto_stop_grace_sec": 8}
    with (
        patch.object(meeting_detector, "_recording_matches", return_value=True),
        patch.object(meeting_detector, "stop_detected_recording", return_value=True) as stop,
        patch.object(meeting_detector, "save_state", side_effect=lambda value: saved.append(dict(value))),
    ):
        assert meeting_detector.handle_auto_recording_lifecycle(
            config,
            {"active": False},
            state,
            100.0,
        ) is False
        assert state["auto_recording"]["missing_since"] == 100.0
        assert meeting_detector.handle_auto_recording_lifecycle(
            config,
            {"active": False},
            state,
            109.0,
        ) is True

    stop.assert_called_once_with(meeting_id)
    assert "auto_recording" not in state
    assert saved


def test_auto_stop_never_stops_a_different_or_manual_recording() -> None:
    state = {
        "auto_recording": {
            "meeting_id": "detected::old",
            "signature": "old",
            "started_at": 1.0,
            "missing_since": 2.0,
        },
    }
    with (
        patch.object(meeting_detector, "_recording_matches", return_value=False),
        patch.object(meeting_detector, "stop_detected_recording") as stop,
        patch.object(meeting_detector, "save_state"),
    ):
        assert meeting_detector.handle_auto_recording_lifecycle(
            meeting_detector.DEFAULT_CONFIG,
            {"active": False},
            state,
            100.0,
        ) is False

    stop.assert_not_called()
    assert "auto_recording" not in state


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
