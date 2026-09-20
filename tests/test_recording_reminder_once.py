# pyright: reportMissingImports=false

from pathlib import Path
from unittest.mock import patch

import meeting_daemon
import scheduler_daemon

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_reminder_ids_are_upserted_instead_of_duplicated(tmp_path: Path) -> None:
    schedule = {
        "events": [
            {"id": "recording-meeting-1::ask_stop", "kind": "ask_stop", "at": "old"},
            {"id": "other", "kind": "remind", "at": "later"},
        ],
        "meetings": [],
    }
    replacement = {
        "id": "recording-meeting-1::ask_stop",
        "kind": "ask_stop",
        "at": "new",
        "meeting_id": "meeting-1",
        "title": "Team Sync",
    }
    with (
        patch.object(meeting_daemon, "SCHEDULE_LOCK_PATH", tmp_path / "schedule.lock"),
        patch.object(meeting_daemon, "load_schedule", return_value=schedule),
        patch.object(meeting_daemon, "save_schedule", side_effect=lambda _data: None),
    ):
        meeting_daemon._add_runtime_event(replacement)

    matching = [event for event in schedule["events"] if event.get("id") == replacement["id"]]
    assert matching == [replacement]


def test_runtime_reminder_can_only_be_claimed_once(tmp_path: Path) -> None:
    event_id = "recording-meeting-1::ask_stop"
    schedule = {
        "events": [{"id": event_id, "kind": "ask_stop", "at": "now"}],
        "meetings": [],
    }
    with (
        patch.object(meeting_daemon, "SCHEDULE_LOCK_PATH", tmp_path / "schedule.lock"),
        patch.object(meeting_daemon, "load_schedule", return_value=schedule),
        patch.object(meeting_daemon, "save_schedule", side_effect=lambda _data: None),
    ):
        assert meeting_daemon._consume_runtime_event(event_id) is True
        assert meeting_daemon._consume_runtime_event(event_id) is False


def test_duplicate_auto_stop_process_does_not_open_another_prompt() -> None:
    with (
        patch.object(meeting_daemon, "_consume_runtime_event", return_value=False),
        patch.object(meeting_daemon.subprocess, "run", side_effect=AssertionError("must not prompt")),
    ):
        meeting_daemon.cmd_auto_stop("recording-meeting-1::ask_stop")


def test_scheduler_passes_the_reminder_id_to_auto_stop() -> None:
    scheduler = scheduler_daemon.Scheduler()
    spawned: list[list[str]] = []
    with patch.object(scheduler, "_spawn", side_effect=lambda command: spawned.append(command)):
        scheduler._fire(
            {
                "id": "recording-meeting-1::ask_stop",
                "kind": "ask_stop",
                "meeting_id": "meeting-1",
                "title": "Team Sync",
            }
        )

    assert spawned == [[
        scheduler_daemon.sys.executable,
        str(scheduler_daemon.SCRIPT_DIR / "meeting_daemon.py"),
        "auto_stop",
        "recording-meeting-1::ask_stop",
    ]]


def test_long_recording_prompt_explains_the_actual_risk() -> None:
    source = (ROOT / "yulu/scripts/meeting_prompt.swift").read_text(encoding="utf-8")
    assert 'L("录音仍在进行", "Recording is still running")' in source
    assert 'L("为避免忘记关闭录音，请确认是否继续。", "To avoid an unintended long recording, confirm whether to continue.")' in source
    assert "会议结束了吗？" not in source
    assert "已到预计结束时间" not in source
