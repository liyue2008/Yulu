import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "yulu" / "scripts"


def test_development_app_owns_components_as_direct_children(tmp_path: Path) -> None:
    binary = tmp_path / "yulu_app"
    built = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "build_yulu_shell.sh"),
            str(binary),
            "-D",
            "YULU_DEVELOPMENT_SMOKE",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert built.returncode == 0, built.stderr

    inspected = subprocess.run(
        [str(binary), "--inspect-component-ownership"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout) == {"mode": "direct_children"}

    source = (SCRIPTS / "yulu_app.swift").read_text(encoding="utf-8")
    startup = source.split(
        "private func startDevelopmentRuntime(applicationPaths:", 1
    )[1].split("#endif", 1)[0]
    assert "ProductSupervisor(" in startup
    assert "supervisor.start()" in startup
    assert "migrationCommitted = true" in startup
    assert 'hostEnvironment["YULU_SERVICE_OWNER"] = "com.yulu.app.host"' in source
    assert 'captureEnvironment["YULU_SERVICE_OWNER"] = "com.yulu.app.capture"' in source
