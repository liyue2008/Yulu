# pyright: reportMissingImports=false

import os
from pathlib import Path

from application_update import _valid_update_health


def test_update_health_accepts_runtime_inside_user_applications() -> None:
    application = Path(os.environ["YULU_REAL_HOME"]) / "Applications/Yulu.app"
    health: dict[str, object] = {
        "application": {
            "identifier": "com.yulu.app",
            "teamIdentifier": "WMU9678ZQL",
            "cdHash": "a" * 40,
            "version": "0.26.0",
            "build": "800",
            "pid": 101,
            "uid": os.geteuid(),
            "generation": "100:1",
            "executable": str(application / "Contents/MacOS/yulu_app"),
            "nativeControlsReady": True,
        },
        "host": {
            "identifier": "node",
            "teamIdentifier": "WMU9678ZQL",
            "cdHash": "b" * 40,
            "productVersion": "0.26.0",
            "bundleVersion": "800",
            "hostIPCVersion": 1,
            "serviceOwner": "com.yulu.app.host",
            "pid": 102,
            "uid": os.geteuid(),
            "generation": "100:2",
            "executable": str(application / "Contents/Resources/runtime/bin/node"),
            "hostNonce": "11111111-1111-4111-8111-111111111111",
            "instanceLockToken": "host-lock-token-1234",
            "portOwnerPID": 102,
            "database": {
                "status": "ok",
                "quickCheck": "ok",
                "schemaVersion": 1,
                "minimumReadableVersion": 1,
            },
        },
        "capture": {
            "identifier": "com.yulu.audiodaemon",
            "teamIdentifier": "WMU9678ZQL",
            "cdHash": "c" * 40,
            "productVersion": "0.26.0",
            "bundleVersion": "800",
            "captureIPCVersion": 1,
            "serviceOwner": "com.yulu.app.capture",
            "pid": 103,
            "uid": os.geteuid(),
            "generation": "100:3",
            "executable": str(
                application
                / "Contents/Helpers/YuluCapture.app/Contents/MacOS/audio_daemon"
            ),
            "socketOwnerPID": 103,
        },
        "services": {
            "com.yulu.ui.plist": "enabled",
            "com.yulu.audiodaemon.plist": "enabled",
        },
    }

    assert _valid_update_health(
        health,
        expected={"version": "0.26.0", "build": "800"},
        application_path=application,
    )
    assert not _valid_update_health(
        health,
        expected={"version": "0.26.0", "build": "800"},
        application_path=Path("/Applications/Yulu.app"),
    )
