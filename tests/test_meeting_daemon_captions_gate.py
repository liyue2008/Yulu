# pyright: reportMissingImports=false

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "yulu" / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(autouse=True)
def no_live_lark_probe(monkeypatch):
    import meeting_daemon

    monkeypatch.setattr(meeting_daemon, "_lark_meeting_is_running", lambda: False)


@pytest.fixture
def config_path(monkeypatch, tmp_path):
    import meeting_daemon

    path = tmp_path / "config.json"
    monkeypatch.setattr(meeting_daemon, "CONFIG_PATH", path)
    # Isolation: never fall back to the real CONFIG_READ_PATHS when the test
    # home has no config of its own.
    monkeypatch.setattr(meeting_daemon, "CONFIG_READ_PATHS", ())
    return path


# ── helpers ──────────────────────────────────────────────────────────────

def test_caption_flags_default_off_when_key_is_missing(config_path):
    import meeting_daemon

    config_path.write_text(json.dumps({"transcription": {}}), encoding="utf-8")
    assert meeting_daemon._captions_realtime_enabled() is False
    assert meeting_daemon._status_window_enabled() is False


def test_caption_flags_default_off_without_any_config_file(monkeypatch, tmp_path):
    import meeting_daemon

    monkeypatch.setattr(meeting_daemon, "CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(meeting_daemon, "CONFIG_READ_PATHS", ())
    assert meeting_daemon._captions_realtime_enabled() is False
    assert meeting_daemon._status_window_enabled() is False


def test_caption_flags_require_explicit_true(config_path):
    import meeting_daemon

    config_path.write_text(json.dumps({"transcription": {"captions": {
        "realtime_enabled": "true",
        "status_window_enabled": 1,
    }}}), encoding="utf-8")
    assert meeting_daemon._captions_realtime_enabled() is False
    assert meeting_daemon._status_window_enabled() is False

    config_path.write_text(json.dumps({"transcription": {"captions": {
        "realtime_enabled": True,
        "status_window_enabled": False,
    }}}), encoding="utf-8")
    assert meeting_daemon._captions_realtime_enabled() is True
    assert meeting_daemon._status_window_enabled() is False

    config_path.write_text(json.dumps({"transcription": {"captions": {
        "status_window_enabled": True,
    }}}), encoding="utf-8")
    assert meeting_daemon._captions_realtime_enabled() is False
    assert meeting_daemon._status_window_enabled() is True


def test_caption_flags_ignore_malformed_config(config_path):
    import meeting_daemon

    config_path.write_text("{not json", encoding="utf-8")
    assert meeting_daemon._captions_realtime_enabled() is False
    assert meeting_daemon._status_window_enabled() is False


# ── start path ───────────────────────────────────────────────────────────

def _prepare_start(monkeypatch, tmp_path):
    import meeting_daemon

    audio = str(tmp_path / "meeting.wav")

    @contextmanager
    def fake_lock(timeout=0.0):
        yield object()

    monkeypatch.setattr(meeting_daemon, "acquire_recording_lock", fake_lock)
    monkeypatch.setattr(
        meeting_daemon, "_daemon_start_recording",
        lambda title, lock_handle=None: audio,
    )
    monkeypatch.setattr(meeting_daemon, "record_lock", lambda *a, **kw: None)
    monkeypatch.setattr(meeting_daemon, "set_recording_started", lambda *a, **kw: None)
    monkeypatch.setattr(meeting_daemon, "_meeting_duration", lambda _mid: 30)
    monkeypatch.setattr(meeting_daemon, "_add_runtime_event", lambda _event: None)
    monkeypatch.setattr(meeting_daemon, "STATE_PATH", tmp_path / ".state.json")

    realtime_calls = []
    monkeypatch.setattr(
        meeting_daemon, "_post_realtime",
        lambda action, payload, **kw: realtime_calls.append((action, payload)) or True,
    )
    window_launches = []
    monkeypatch.setattr(
        meeting_daemon, "_launch_status_window",
        lambda title: window_launches.append(title),
    )
    return meeting_daemon, realtime_calls, window_launches


def test_start_skips_realtime_and_window_when_flags_missing(monkeypatch, tmp_path, config_path):
    meeting_daemon, realtime_calls, window_launches = _prepare_start(monkeypatch, tmp_path)
    config_path.write_text(json.dumps({"transcription": {}}), encoding="utf-8")

    assert meeting_daemon._start_recording("Team Sync") is True
    assert realtime_calls == []
    assert window_launches == []


def test_start_posts_realtime_and_launches_window_when_enabled(monkeypatch, tmp_path, config_path):
    meeting_daemon, realtime_calls, window_launches = _prepare_start(monkeypatch, tmp_path)
    config_path.write_text(json.dumps({"transcription": {"captions": {
        "realtime_enabled": True,
        "status_window_enabled": True,
    }}}), encoding="utf-8")

    assert meeting_daemon._start_recording("Team Sync") is True
    assert [action for action, _payload in realtime_calls] == ["start"]
    assert realtime_calls[0][1]["title"] == "Team Sync"
    assert window_launches == ["Team Sync"]


def test_start_enables_realtime_and_window_independently(monkeypatch, tmp_path, config_path):
    meeting_daemon, realtime_calls, window_launches = _prepare_start(monkeypatch, tmp_path)
    config_path.write_text(json.dumps({"transcription": {"captions": {
        "realtime_enabled": True,
    }}}), encoding="utf-8")

    assert meeting_daemon._start_recording("Team Sync") is True
    assert [action for action, _payload in realtime_calls] == ["start"]
    assert window_launches == []


# ── stop path ────────────────────────────────────────────────────────────

def _prepare_stop(monkeypatch, tmp_path):
    import meeting_daemon

    wav = tmp_path / "TeamSync_20260102_090000.wav"
    wav.write_bytes(b"RIFFxxxxWAVE")

    monkeypatch.setattr(meeting_daemon, "_active_recording_info", lambda: {
        "title": "Team Sync",
        "audio_path": str(wav),
        "file_path": str(wav),
        "backend": "daemon",
        "transcription_language": "zh",
    })
    monkeypatch.setattr(meeting_daemon, "set_recording_stopped", lambda **_kw: {})
    monkeypatch.setattr(meeting_daemon, "load_schedule", lambda: {"events": [], "meetings": []})
    monkeypatch.setattr(meeting_daemon, "save_schedule", lambda _data: None)
    monkeypatch.setattr(
        meeting_daemon.subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=0,
            stdout=f"FINAL_RECORDING_PATH={wav}\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(meeting_daemon, "MCP_TOKEN_PATH", tmp_path / "mcp-token.json")
    monkeypatch.setattr(meeting_daemon, "RECORDING_EVENTS_DIR", tmp_path / "recording-events")
    monkeypatch.setattr(
        meeting_daemon, "_dispatch_recording_completed",
        lambda *_a, **_kw: "accepted",
    )

    realtime_calls = []
    monkeypatch.setattr(
        meeting_daemon, "_post_realtime",
        lambda action, payload, **kw: realtime_calls.append((action, payload)) or True,
    )
    return meeting_daemon, realtime_calls


def test_stop_skips_realtime_post_when_flag_missing(monkeypatch, tmp_path, config_path):
    meeting_daemon, realtime_calls = _prepare_stop(monkeypatch, tmp_path)
    config_path.write_text(json.dumps({"transcription": {}}), encoding="utf-8")

    assert meeting_daemon._stop_and_process() is True
    assert realtime_calls == []


def test_stop_posts_realtime_stop_when_enabled(monkeypatch, tmp_path, config_path):
    meeting_daemon, realtime_calls = _prepare_stop(monkeypatch, tmp_path)
    config_path.write_text(json.dumps({"transcription": {"captions": {
        "realtime_enabled": True,
    }}}), encoding="utf-8")

    assert meeting_daemon._stop_and_process() is True
    assert [action for action, _payload in realtime_calls] == ["stop"]


def test_stop_reports_realtime_failure_when_enabled(monkeypatch, tmp_path, config_path, capsys):
    meeting_daemon, realtime_calls = _prepare_stop(monkeypatch, tmp_path)
    monkeypatch.setattr(meeting_daemon, "_post_realtime", lambda *a, **kw: False)
    config_path.write_text(json.dumps({"transcription": {"captions": {
        "realtime_enabled": True,
    }}}), encoding="utf-8")

    assert meeting_daemon._stop_and_process() is True
    assert "实时转写收尾失败" in capsys.readouterr().err
