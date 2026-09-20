import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "yulu" / "scripts"


def test_shell_accepts_system_and_user_applications_directories(tmp_path: Path) -> None:
    binary = tmp_path / "yulu_app"
    compiled = subprocess.run(
        [
            "swiftc",
            "-module-cache-path",
            str(tmp_path / "swift-cache"),
            "-o",
            str(binary),
            str(SCRIPTS / "yulu_app.swift"),
            "-framework",
            "Cocoa",
            "-framework",
            "WebKit",
            "-framework",
            "ServiceManagement",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr

    def inspect(command: str, path: Path) -> dict[str, object]:
        result = subprocess.run(
            [str(binary), command, str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    real_home = Path(os.environ["YULU_REAL_HOME"])
    supported = [Path("/Applications/Yulu.app"), real_home / "Applications/Yulu.app"]
    expected_launch = {
        "installed": True,
        "persistentRegistrationAllowed": True,
        "componentsStarted": True,
        "guidance": None,
    }
    expected_actions = {
        "persistentFileWrites": [],
        "register": ["com.yulu.ui.plist", "com.yulu.audiodaemon.plist"],
        "unregister": [],
    }
    for path in supported:
        assert inspect("--inspect-launch", path) == expected_launch
        assert inspect("--inspect-service-actions", path) == expected_actions

    for path in (Path("/Volumes/Yulu/Yulu.app"), real_home / "Downloads/Yulu.app"):
        assert inspect("--inspect-launch", path) == {
            "installed": False,
            "persistentRegistrationAllowed": False,
            "componentsStarted": False,
            "guidance": "Move Yulu to /Applications or ~/Applications before opening it.",
        }
        assert inspect("--inspect-service-actions", path) == {
            "persistentFileWrites": [],
            "register": [],
            "unregister": [],
        }
