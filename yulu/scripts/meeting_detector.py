#!/usr/bin/env python3
"""
会议检测守护进程：检测常见会议/通话窗口，触发录制询问。

目标：不依赖日历，发现微信语音/视频电话、腾讯会议、Google Meet、飞书/Lark、Zoom 等会议场景。

策略：
- 优先读取 macOS 前台应用/窗口标题（System Events）
- 用窗口标题关键词判定会议状态，避免仅因微信/浏览器常驻而误报
- 检测持续 stable_sec 后才弹窗
- 同一 meeting signature 有 cooldown，避免重复询问
- 已在录制时不弹窗
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from application_paths import (
    CONFIG_PATH,
    CONFIG_READ_PATHS,
    DURABLE_DATA_DIR,
    IPC_DIR,
    LEGACY_READ_ONLY_DATA_DIR,
    LOGS_DIR,
)
from state_store import is_recording_active as state_recording_active
from state_store import load_state as load_recording_state
from state_store import recording_info

CONFIG_DIR = DURABLE_DATA_DIR
STATE_PATH = CONFIG_DIR / ".detector_state.json"
RECORDING_STATE_PATH = CONFIG_DIR / ".state.json"
PID_PATH = IPC_DIR / ".detector.pid"
LOG_PATH = LOGS_DIR / "detector.log"
SCRIPT_DIR = Path(__file__).resolve().parent
_LARK_CLI_USER_OPEN_IDS = {}
LARK_RTC_LOG_DIRS = (
    Path.home() / "Library/Application Support/LarkInternational/sdk_storage/log/native-pc-sdk/rtc-sdk",
    Path.home() / "Library/Application Support/Lark/sdk_storage/log/native-pc-sdk/rtc-sdk",
    Path.home() / "Library/Application Support/Feishu/sdk_storage/log/native-pc-sdk/rtc-sdk",
)
LARK_RTC_LOG_FRESHNESS_SEC = 15
LARK_RTC_LOG_SCAN_BYTES = 1024 * 1024

DEFAULT_CONFIG = {
    "enabled": True,
    "interval_sec": 10,
    "stable_sec": 15,
    "prompt_cooldown_sec": 1800,
    "window_keywords": [
        # Zoom
        "Zoom Meeting", "Zoom 会议", "zoom.us", "Zoom Workplace",
        # Tencent Meeting / VooV
        "腾讯会议", "Tencent Meeting", "VooV Meeting", "WeMeet",
        # Google Meet (browser windows)
        "Google Meet", "meet.google.com", "Meet -",
        # Feishu / Lark
        "飞书会议", "Feishu Meeting", "Lark Meeting", "Lark | Meeting", "视频会议",
        # WeChat / WeCom calls
        "微信通话", "微信电话", "语音通话", "视频通话", "Voice Call", "Video Call",
        "WeChat Call", "WeChat Video", "企业微信", "WeCom",
        # Generic browser/app meeting hints
        "正在通话", "正在会议", "加入会议", "会议中", "通话中",
    ],
    "app_name_hints": [
        "zoom.us", "Zoom", "腾讯会议", "TencentMeeting", "VooV", "WeMeet",
        "Feishu", "Lark", "飞书", "WeChat", "微信", "企业微信", "WeCom",
        "Google Chrome", "Arc", "Safari", "Microsoft Edge", "Firefox",
    ],
    "target_app_names": [
        "zoom.us", "Zoom", "腾讯会议", "TencentMeeting", "VooV Meeting", "WeMeet",
        "Feishu", "Lark", "飞书", "WeChat", "微信", "企业微信", "WeCom",
        "Google Chrome", "Arc", "Safari", "Microsoft Edge", "Firefox"
    ],
    "dedicated_meeting_apps": [
        "zoom.us", "Zoom", "腾讯会议", "TencentMeeting", "VooV Meeting", "WeMeet"
    ],
    "process_app_names": [
        "LarkSuite.app", "/Lark", "Feishu",
    ],
    "meeting_process_keywords": [
        "byteview", "--scene=vc", "--scene=meeting",
        "--module=renderer-byteview", "--module=renderer-meeting",
    ],
    "auto_record_apps": [],
    "lark_cli_active_meeting": False,
    "auto_stop_grace_sec": 10,
    "ignore_window_keywords": [
        "Calendar", "日历", "Gmail", "Inbox", "Settings", "Preferences",
        "聊天", "通讯录", "朋友圈", "文件传输助手",
        "PolyMeet",
    ],
}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


AUDIO_DAEMON_SOCKET = IPC_DIR / "audio_daemon.sock"
WINDOW_SCANNER = SCRIPT_DIR / "window_scanner"


def _query_audio_daemon():
    """通过 Yulu 的 socket 获取窗口列表。"""
    if not AUDIO_DAEMON_SOCKET.exists():
        return None
    try:
        import socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(str(AUDIO_DAEMON_SOCKET))
        sock.sendall(b'{"action":"windows"}')
        sock.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        sock.close()
        resp = json.loads(data.decode())
        return resp.get("windows", [])
    except Exception:
        return None


def _has_permission():
    """Quick check: can we read window titles?"""
    # Try Yulu first (merged scanner)
    wins = _query_audio_daemon()
    if wins is not None:
        return True
    # Fallback: standalone window_scanner
    try:
        r = subprocess.run(
            [str(WINDOW_SCANNER)],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False


def collect_windows(target_app_names=None):
    """返回 [{app, title}] 或 None。
    优先用 Yulu 的 socket（如运行中），否则用 window_scanner。"""
    windows = _query_audio_daemon()
    if windows is None:
        try:
            r = subprocess.run(
                [str(WINDOW_SCANNER)],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return None
            windows = json.loads(r.stdout)
        except Exception:
            return None

    if not windows:
        return []

    target_app_names = target_app_names or DEFAULT_CONFIG["target_app_names"]
    target_patterns = _compile_patterns(target_app_names)

    rows = []
    for w in windows:
        app = w.get("app", "")
        title = w.get("title", "")
        if not app or not title:
            continue
        if not any(p.search(app) for p in target_patterns):
            continue
        rows.append({"app": app.strip(), "title": title.strip()})
    return rows


def load_config():
    path = CONFIG_PATH if CONFIG_PATH.exists() else next(
        (candidate for candidate in CONFIG_READ_PATHS if candidate.exists()), None
    )
    if path is None:
        return DEFAULT_CONFIG.copy()
    try:
        with path.open() as f:
            full = json.load(f)
    except Exception:
        return DEFAULT_CONFIG.copy()
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(full.get("meeting_detection", {}))
    return cfg


def load_state():
    path = STATE_PATH
    if not path.exists():
        legacy = LEGACY_READ_ONLY_DATA_DIR / ".detector_state.json"
        if legacy.exists():
            path = legacy
    if not path.exists():
        return {"prompted": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"prompted": {}}


def save_state(state):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _capture_status():
    try:
        from record_audio import _capture_controller
        status = _capture_controller().status()
    except Exception:
        return {}
    return status if isinstance(status, dict) else {}


def _same_audio_path(left, right):
    if not left or not right:
        return False
    try:
        return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()
    except (OSError, RuntimeError):
        return False


def is_recording_active():
    capture = _capture_status()
    recording = capture.get("recording")
    if type(recording) is bool:
        return recording
    try:
        state = load_recording_state(RECORDING_STATE_PATH)
        if state_recording_active(state):
            return True
    except Exception:
        return (IPC_DIR / ".recording_pid").exists()
    return (IPC_DIR / ".recording_pid").exists()


def _osascript(script, timeout=3):
    return subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def collect_visible_apps():
    """无需辅助功能权限的兜底：只能看到可见 app 名，不能看到窗口标题。"""
    result = subprocess.run(["lsappinfo", "visibleProcessList"], capture_output=True, text=True, timeout=5)
    apps = re.findall(r'\-"([^"]+)"', result.stdout)
    return [a.replace("_", " ") for a in apps]


def collect_running_process_commands():
    """Return process command lines without requiring Accessibility or admin."""
    result = subprocess.run(
        ["/bin/ps", "-ww", "-axo", "command="],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError("process enumeration failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _lark_cli_executable():
    candidates = [
        shutil.which("lark-cli"),
        str(Path.home() / ".npm-global/bin/lark-cli"),
        str(Path.home() / ".local/bin/lark-cli"),
        str(Path.home() / "bin/lark-cli"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            executable = Path(candidate).is_file() and os.access(candidate, os.X_OK)
        except OSError:
            executable = False
        if executable:
            return candidate
    return None


def _run_lark_cli_json(arguments, timeout=8):
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    if result.returncode != 0 or len(stdout.encode("utf-8")) > 1024 * 1024:
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _lark_cli_user_open_id(executable):
    cached = _LARK_CLI_USER_OPEN_IDS.get(executable)
    if cached:
        return cached
    payload = _run_lark_cli_json([executable, "auth", "status"])
    identities = payload.get("identities") if isinstance(payload, dict) else None
    user = identities.get("user") if isinstance(identities, dict) else None
    open_id = str(user.get("openId", "")).strip() if isinstance(user, dict) else ""
    if open_id:
        _LARK_CLI_USER_OPEN_IDS[executable] = open_id
    return open_id or None


def _lark_cli_current_user_is_joined(executable, meeting_id):
    open_id = _lark_cli_user_open_id(executable)
    if not open_id:
        return False
    payload = _run_lark_cli_json([
        executable,
        "vc",
        "meeting",
        "get",
        "--as",
        "user",
        "--meeting-id",
        meeting_id,
        "--with-participants",
        "--user-id-type",
        "open_id",
        "--format",
        "json",
    ])
    data = payload.get("data") if isinstance(payload, dict) else None
    meeting = data.get("meeting") if isinstance(data, dict) else None
    participants = meeting.get("participants") if isinstance(meeting, dict) else None
    if not isinstance(participants, list):
        return False
    return any(
        isinstance(participant, dict)
        and str(participant.get("id", "")).strip() == open_id
        and str(participant.get("status", "")).strip() == "2"
        for participant in participants
    )


def _lark_local_rtc_is_joined(meeting_id, now=None):
    if not meeting_id:
        return False
    current_time = time.time() if now is None else now
    candidates = []
    for directory in LARK_RTC_LOG_DIRS:
        try:
            candidates.extend(directory.glob("*_rtclog.log"))
        except OSError:
            continue
    fresh_logs = []
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        age = current_time - stat.st_mtime
        if -5 <= age <= LARK_RTC_LOG_FRESHNESS_SEC:
            fresh_logs.append((stat.st_mtime, stat.st_size, path))
    room_markers = (
        f"room_id:{meeting_id}".encode(),
        f"room_id_{meeting_id}".encode(),
    )
    for _mtime, size, path in sorted(fresh_logs, reverse=True)[:3]:
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, size - LARK_RTC_LOG_SCAN_BYTES))
                chunk = handle.read(LARK_RTC_LOG_SCAN_BYTES)
        except OSError:
            continue
        if any(marker in chunk for marker in room_markers):
            return True
    return False


def _detect_lark_cli_meeting(cfg):
    enabled = cfg.get("lark_cli_active_meeting")
    if type(enabled) is not bool or not enabled:
        return None
    executable = _lark_cli_executable()
    if not executable:
        return None
    payload = _run_lark_cli_json([
        executable,
        "vc",
        "+meeting-list-active",
        "--as",
        "user",
        "--format",
        "json",
    ])
    data = payload.get("data") if isinstance(payload, dict) else None
    meetings = data.get("meetings") if isinstance(data, dict) else None
    if not isinstance(meetings, list):
        return None
    for meeting in meetings:
        if not isinstance(meeting, dict):
            continue
        meeting_id = str(meeting.get("meeting_id", "")).strip()
        if not meeting_id:
            continue
        if not _lark_cli_current_user_is_joined(executable, meeting_id):
            continue
        if not _lark_local_rtc_is_joined(meeting_id):
            continue
        title = normalize_title(str(meeting.get("meeting_title", "")).strip() or "Lark Meeting")
        return {
            "active": True,
            "title": title,
            "app": "Lark",
            "window": "",
            "signature": signature("Lark", meeting_id),
            "fallback": "lark_cli",
        }
    return None


def _compile_patterns(words):
    return [re.compile(re.escape(w), re.I) for w in words if w]


def _detect_meeting_process(cfg):
    try:
        commands = collect_running_process_commands()
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None
    app_patterns = _compile_patterns(cfg.get("process_app_names", []))
    meeting_patterns = _compile_patterns(cfg.get("meeting_process_keywords", []))
    if not app_patterns or not meeting_patterns:
        return None
    for command in commands:
        if not any(pattern.search(command) for pattern in app_patterns):
            continue
        if not any(pattern.search(command) for pattern in meeting_patterns):
            continue
        app = "Lark" if re.search(r"Lark|Feishu", command, re.I) else "Meeting app"
        return {
            "active": True,
            "title": f"{app} Meeting",
            "app": app,
            "window": "",
            "signature": signature(app, "meeting-process"),
            "fallback": "meeting_process",
        }
    return None


def _detect_running_meeting_app(cfg, ignore_patterns, window_patterns):
    """Detect a launched meeting app without requesting Accessibility access."""
    try:
        running_apps = collect_visible_apps()
    except Exception:
        return None

    dedicated = _compile_patterns(cfg.get("dedicated_meeting_apps", []))
    for app in running_apps:
        if any(p.search(app) for p in dedicated):
            return {
                "active": True,
                "title": app,
                "app": app,
                "window": "",
                "signature": signature(app, "running-process"),
                "fallback": "running_process",
            }

    for app in running_apps:
        if any(p.search(app) for p in ignore_patterns):
            continue
        if any(p.search(app) for p in window_patterns):
            return {
                "active": True,
                "title": app,
                "app": app,
                "window": "",
                "signature": signature(app, "running-process-keyword"),
                "fallback": "running_process",
            }
    return None


def detect_meeting(cfg):
    """返回检测结果或 None。"""
    lark_meeting = _detect_lark_cli_meeting(cfg)
    if lark_meeting is not None:
        return lark_meeting

    window_patterns = _compile_patterns(cfg.get("window_keywords", []))
    ignore_patterns = _compile_patterns(cfg.get("ignore_window_keywords", []))
    app_hints = _compile_patterns(cfg.get("app_name_hints", []))

    windows = collect_windows(cfg.get("target_app_names")) or []
    matches = []
    for w in windows:
        haystack = f"{w['app']} {w['title']}"
        if any(p.search(haystack) for p in ignore_patterns):
            continue
        app_interesting = any(p.search(w["app"]) for p in app_hints)
        title_match = any(p.search(w["title"]) or p.search(haystack) for p in window_patterns)
        if title_match and app_interesting:
            matches.append(w)

    if matches:
        best = sorted(matches, key=lambda x: len(x.get("title", "")), reverse=True)[0]
        title = best.get("title") or best.get("app") or "检测到会议"
        return {
            "active": True,
            "title": normalize_title(title),
            "app": best.get("app", ""),
            "window": best.get("title", ""),
            "signature": signature(best.get("app", ""), title),
            "matches": matches[:5],
        }

    process_match = _detect_meeting_process(cfg)
    if process_match is not None:
        return process_match

    running_match = _detect_running_meeting_app(cfg, ignore_patterns, window_patterns)
    if running_match is not None:
        return running_match
    return {"active": False, "windows": windows[:10]}


# macOS 14+ 会把动态状态短语塞进浏览器窗口标题末尾（"麦克风正在录音"、
# "内存用量高 - 811 MB" 等），还带 "- Google Chrome - <profile>" 尾巴。这些
# 每隔几秒就变，导致 signature 漂移、detector 永远凑不满 stable_sec、
# `🔔 提醒录制` 不触发。剥掉它们让同一会议产生稳定 signature。
_SYSTEM_STATUS_PATTERNS = [
    re.compile(r"\s*[-–—]\s*摄像头正在录像且麦克风正在录音"),
    re.compile(r"\s*[-–—]\s*麦克风正在录音"),
    re.compile(r"\s*[-–—]\s*摄像头正在录像"),
    re.compile(r"\s*[-–—]\s*已分享桌面内容"),
    re.compile(r"\s*[-–—]\s*正在共享屏幕"),
    re.compile(r"\s*[-–—]\s*内存用量高\s*[-–—]\s*[\d.,]+\s*(?:KB|MB|GB|TB)", re.IGNORECASE),
    re.compile(r"\s*[-–—]\s*Audio is playing", re.IGNORECASE),
    re.compile(r"\s*[-–—]\s*Camera (?:and microphone )?is on", re.IGNORECASE),
    re.compile(r"\s*[-–—]\s*Microphone is on", re.IGNORECASE),
    re.compile(r"\s*[-–—]\s*Screen sharing", re.IGNORECASE),
    re.compile(r"\s*[-–—]\s*High memory usage\s*[-–—]\s*[\d.,]+\s*(?:KB|MB|GB|TB)", re.IGNORECASE),
]

_BROWSER_TAIL_PATTERN = re.compile(
    r"\s*[-–—]\s*(?:Google Chrome|Chrome|Arc|Safari|Microsoft Edge|Edge|Firefox|Brave|Vivaldi|Opera)\b.*$",
    re.IGNORECASE,
)

# Chrome 不在前台时窗口标题会丢掉 "- Google Chrome"，只剩 "- <profile>"，
# 例如 "Meet - tcu-oyza-tje - Bill"。这种情况 _BROWSER_TAIL_PATTERN 不命中，
# 兜底剥一次单段尾巴。只匹配 ASCII profile（Chrome 默认 profile 名都是
# 英文："Bill" / "Personal" / "Work"），避免误伤中文会议名末段如
# "腾讯会议 - 周会"。
_TRAILING_PROFILE_PATTERN = re.compile(
    r"\s*[-–—]\s*[A-Za-z][A-Za-z0-9_.\-]{0,19}\s*$"
)


def strip_system_status(title):
    """剥掉 macOS 在标题末尾注入的浏览器状态短语和浏览器/profile 尾巴，
    保留会议本身的标题。让同一会议在 detector 看来 signature 稳定。"""
    if not title:
        return title
    prev = None
    # 反复跑直到没有可剥的，应对多状态叠加（如 麦克风 + 内存）
    while prev != title:
        prev = title
        for pat in _SYSTEM_STATUS_PATTERNS:
            title = pat.sub("", title)
    # 媒体状态短语剥光后，再砍掉浏览器/profile 尾巴。
    new_title = _BROWSER_TAIL_PATTERN.sub("", title)
    if new_title == title:
        # 浏览器名缺失（窗口失焦），用 profile 兜底再剥一次单段尾巴。
        new_title = _TRAILING_PROFILE_PATTERN.sub("", title)
    return new_title.rstrip(" -–—")


def normalize_title(title):
    title = strip_system_status(title)
    title = re.sub(r"\s+", " ", title).strip()
    title = title[:80]
    return title or "检测到会议"


def signature(app, title):
    raw = f"{app}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:12]


def recently_prompted(state, sig, cooldown_sec):
    prompted = state.setdefault("prompted", {})
    now = time.time()
    # 顺手清理过期记录
    for key, ts in list(prompted.items()):
        try:
            expired = now - float(ts) > cooldown_sec * 2
        except (TypeError, ValueError):
            expired = True
        if expired:
            prompted.pop(key, None)
    if sig not in prompted:
        return False
    try:
        return now - float(prompted[sig]) < cooldown_sec
    except (TypeError, ValueError):
        prompted.pop(sig, None)
        return False


def mark_prompted(state, sig):
    state.setdefault("prompted", {})[sig] = time.time()
    save_state(state)


def _launch_recording_command(arguments):
    daemon = SCRIPT_DIR / "meeting_daemon.py"
    subprocess.Popen(
        [sys.executable, str(daemon), *arguments],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.environ.get("YULU_MANAGED_REMINDERS") != "1",
    )


def prompt_recording(title):
    _launch_recording_command([
        "ask_record",
        title,
        f"detected::{signature('', title)}",
    ])


def recording_title(result):
    title = normalize_title(str(result.get("title", "")).strip() or "会议")
    if result.get("fallback") == "lark_cli":
        return title
    return f"检测到会议：{title}"


def detected_meeting_id(result):
    detected_signature = str(result.get("signature", "")).strip()
    return f"detected::{detected_signature}" if detected_signature else ""


def _recording_matches(meeting_id, audio_path=""):
    try:
        current = recording_info(load_recording_state(RECORDING_STATE_PATH))
    except Exception:
        current = None
    capture = _capture_status()
    recording = capture.get("recording")
    if type(recording) is bool:
        if not recording:
            return False
        live_path = capture.get("file") or ""
        if audio_path and live_path and not _same_audio_path(audio_path, live_path):
            return False
        if current and current.get("meeting_id") == meeting_id:
            return True
        return bool(audio_path and _same_audio_path(audio_path, live_path))
    return bool(current and current.get("meeting_id") == meeting_id)


def _detected_recording_info(meeting_id):
    try:
        current = recording_info(load_recording_state(RECORDING_STATE_PATH))
    except Exception:
        return None
    if not current or current.get("meeting_id") != meeting_id:
        return None
    audio_path = current.get("audio_path") or current.get("file_path") or ""
    if not audio_path or not _recording_matches(meeting_id, audio_path):
        return None
    return current


def _start_detected_recording(title, meeting_id):
    daemon = SCRIPT_DIR / "meeting_daemon.py"
    with suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, str(daemon), "start", title, meeting_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=45,
        )
    return _detected_recording_info(meeting_id)


def dispatch_recording(cfg, result, title):
    auto_patterns = _compile_patterns(cfg.get("auto_record_apps", []))
    app = str(result.get("app", ""))
    meeting_id = detected_meeting_id(result)
    if meeting_id and auto_patterns and any(pattern.search(app) for pattern in auto_patterns):
        started = _start_detected_recording(title, meeting_id)
        return ("automatic", started) if started else ("failed", None)
    prompt_recording(title)
    return "prompt", None


def stop_detected_recording(meeting_id, audio_path=""):
    daemon = SCRIPT_DIR / "meeting_daemon.py"
    arguments = [sys.executable, str(daemon), "stop_detected", meeting_id]
    if audio_path:
        arguments.append(str(audio_path))
    try:
        result = subprocess.run(
            arguments,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def handle_auto_recording_lifecycle(cfg, result, state, now):
    marker = state.get("auto_recording")
    if not isinstance(marker, dict):
        return False
    meeting_id = marker.get("meeting_id")
    detected_signature = marker.get("signature")
    if not isinstance(meeting_id, str) or not meeting_id.startswith("detected::"):
        state.pop("auto_recording", None)
        save_state(state)
        return False
    audio_path = marker.get("audio_path")
    expected_audio_path = str(audio_path).strip() if isinstance(audio_path, str) else ""
    recording_matches = _recording_matches(meeting_id, expected_audio_path)
    if result.get("active") and result.get("signature") == detected_signature:
        started_at = marker.get("started_at")
        try:
            start_age = (
                now - float(started_at)
                if isinstance(started_at, (int, float, str))
                else 0
            )
        except ValueError:
            start_age = 0
        if not recording_matches and start_age > 15:
            state.pop("auto_recording", None)
            save_state(state)
        elif "missing_since" in marker:
            marker.pop("missing_since", None)
            save_state(state)
        return False
    if not recording_matches:
        state.pop("auto_recording", None)
        save_state(state)
        return False
    missing_since = marker.get("missing_since")
    if missing_since is None:
        marker["missing_since"] = now
        save_state(state)
        return False
    try:
        if not isinstance(missing_since, (int, float, str)):
            raise ValueError("invalid missing_since")
        missing_seconds = now - float(missing_since)
        grace_value = cfg.get("auto_stop_grace_sec", 10)
        if not isinstance(grace_value, (int, float, str)):
            raise ValueError("invalid auto_stop_grace_sec")
        grace_seconds = max(0, int(grace_value))
    except (TypeError, ValueError):
        marker["missing_since"] = now
        save_state(state)
        return False
    if missing_seconds < grace_seconds:
        return False
    if not stop_detected_recording(meeting_id, expected_audio_path):
        return False
    state.pop("auto_recording", None)
    save_state(state)
    log(f"⏹ automatic stop: {meeting_id}")
    return True


def run_once(args):
    cfg = load_config()
    res = detect_meeting(cfg)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res.get("active") else 1


def run_daemon(args):
    cfg = load_config()
    if not cfg.get("enabled", True):
        log("meeting_detection disabled")
        return

    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))
    try:
        interval = max(1, int(cfg.get("interval_sec", 10)))
        stable_sec = max(0, int(cfg.get("stable_sec", 15)))
        cooldown = max(0, int(cfg.get("prompt_cooldown_sec", 1800)))
    except (TypeError, ValueError):
        interval = 10
        stable_sec = 15
        cooldown = 1800
        log("invalid meeting detection timing config; using defaults")

    state = load_state()
    active_since = None
    active_sig = None
    last_permission_error = 0

    log(f"meeting detector started (interval={interval}s stable={stable_sec}s cooldown={cooldown}s)")
    try:
        while True:
            cfg = load_config()
            if not cfg.get("enabled", True):
                time.sleep(interval)
                continue

            res = detect_meeting(cfg)
            now = time.time()
            state = load_state()
            if handle_auto_recording_lifecycle(cfg, res, state, now):
                active_since = None
                active_sig = None
                time.sleep(interval)
                continue

            hint = res.get("permission_hint")
            if hint and now - last_permission_error > 300:
                log(f"⚠️ {hint}")
                last_permission_error = now

            if not res.get("active"):
                active_since = None
                active_sig = None
                time.sleep(interval)
                continue

            sig = res.get("signature")
            if sig != active_sig:
                active_sig = sig
                active_since = now
                log(f"👀 detected candidate: {res.get('app')} | {res.get('window')}")
                time.sleep(interval)
                continue

            if now - (active_since or now) < stable_sec:
                time.sleep(interval)
                continue

            if is_recording_active():
                time.sleep(interval)
                continue

            if recently_prompted(state, sig, cooldown):
                time.sleep(interval)
                continue

            title = recording_title(res)
            mark_prompted(state, sig)
            dispatch, started_recording = dispatch_recording(cfg, res, title)
            if dispatch == "automatic" and started_recording:
                state["auto_recording"] = {
                    "meeting_id": detected_meeting_id(res),
                    "signature": sig,
                    "started_at": now,
                    "audio_path": started_recording.get("audio_path")
                        or started_recording.get("file_path")
                        or "",
                }
                save_state(state)
            log(f"🔔 {dispatch} recording: {title}")
            time.sleep(interval)
    finally:
        with suppress(Exception):
            PID_PATH.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Detect active meeting/call windows and prompt recording")
    parser.add_argument("command", nargs="?", default="daemon", choices=["daemon", "once"])
    args = parser.parse_args()
    if args.command == "once":
        raise SystemExit(run_once(args))
    run_daemon(args)


if __name__ == "__main__":
    main()
