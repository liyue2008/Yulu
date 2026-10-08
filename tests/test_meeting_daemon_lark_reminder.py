# pyright: reportMissingImports=false

from types import SimpleNamespace

import meeting_daemon
import meeting_detector
import pytest


@pytest.fixture
def reminder(monkeypatch):
    state = {
        "title": "Team Sync", "meeting_id": "manual",
        "audio_path": "/test/recording.wav", "started_at": "before",
    }
    events, prompts = [], []
    monkeypatch.setattr(meeting_daemon, "load_state", lambda: state.copy())
    monkeypatch.setattr(meeting_daemon, "recording_info", lambda value: value)
    monkeypatch.setattr(meeting_daemon, "_add_runtime_event", events.append)
    monkeypatch.setattr(meeting_daemon, "_stop_and_process", lambda **_kw: pytest.fail("must not stop"))

    def prompt(*args, **_kwargs):
        prompts.append(args)
        return SimpleNamespace(stdout="继续录制")

    monkeypatch.setattr(meeting_daemon.subprocess, "run", prompt)
    return state, events, prompts


@pytest.mark.parametrize("meeting_id", ["manual", "detected::meeting-1"])
def test_active_lark_meeting_skips_prompt_and_rechecks_in_30_minutes(reminder, monkeypatch, meeting_id):
    state, events, prompts = reminder
    state["meeting_id"] = meeting_id
    monkeypatch.setattr(meeting_daemon, "_lark_meeting_is_running", lambda: True)
    before = meeting_daemon.datetime.now()

    meeting_daemon.cmd_auto_stop()

    assert prompts == []
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "ask_stop"
    assert event["id"] == f"recording-{meeting_id}::ask_stop_extended"
    assert event["meeting_id"] == meeting_id
    assert event["title"] == state["title"]
    delay = (meeting_daemon.datetime.fromisoformat(event["at"]) - before).total_seconds()
    assert 1800 <= delay <= 1805


def test_reminder_returns_after_lark_meeting_ends(reminder, monkeypatch):
    _state, events, prompts = reminder
    active = iter([True, False])
    monkeypatch.setattr(meeting_daemon, "_lark_meeting_is_running", lambda: next(active))

    meeting_daemon.cmd_auto_stop()
    meeting_daemon.cmd_auto_stop()

    assert len(prompts) == 1
    assert "ask_stop" in prompts[0][0]
    assert len(events) == 2


@pytest.mark.parametrize("lark_running", [True, False])
def test_recording_changed_during_lark_probe_does_not_prompt_or_reschedule(reminder, monkeypatch, lark_running):
    state, events, prompts = reminder
    snapshots = iter([state.copy(), {**state, "started_at": "after"}])
    monkeypatch.setattr(meeting_daemon, "load_state", lambda: next(snapshots))
    monkeypatch.setattr(meeting_daemon, "_lark_meeting_is_running", lambda: lark_running)

    meeting_daemon.cmd_auto_stop()

    assert prompts == []
    assert events == []


def test_no_recording_does_not_query_lark(reminder, monkeypatch):
    _state, events, prompts = reminder
    monkeypatch.setattr(meeting_daemon, "recording_info", lambda _state: None)
    monkeypatch.setattr(meeting_daemon, "_lark_meeting_is_running", lambda: pytest.fail("must not query"))

    meeting_daemon.cmd_auto_stop()

    assert events == []
    assert prompts == []


@pytest.mark.parametrize("result, expected", [(None, False), ({"active": True}, True)])
def test_lark_probe_reuses_joined_user_and_local_rtc_checks(monkeypatch, result, expected):
    calls = []

    def probe(config):
        calls.append(config)
        return result

    monkeypatch.setattr(meeting_detector, "_detect_lark_cli_meeting", probe)
    assert meeting_daemon._lark_meeting_is_running() is expected
    # Reminder suppression does not depend on the automatic-start setting.
    assert calls == [{"lark_cli_active_meeting": True}]


def test_lark_probe_failure_keeps_the_reminder(reminder, monkeypatch):
    _state, events, prompts = reminder

    def unavailable(_config):
        raise meeting_daemon.subprocess.TimeoutExpired("lark-cli", 8)

    monkeypatch.setattr(meeting_detector, "_detect_lark_cli_meeting", unavailable)
    meeting_daemon.cmd_auto_stop()

    assert len(prompts) == 1
    assert len(events) == 1
