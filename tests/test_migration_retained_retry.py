# pyright: reportMissingImports=false

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from application_migration import (
    ApplicationMigration,
    MigrationBlocked,
    MigrationPaths,
    _retained_runtime_retry_transaction,
    preflight_standard_outputs,
)


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    node = tmp_path / "Yulu.app/Contents/Resources/runtime/bin/node"
    server = tmp_path / "Yulu.app/Contents/Resources/Host/server.js"
    node.parent.mkdir(parents=True)
    server.parent.mkdir(parents=True)
    node.write_bytes(b"node")
    server.write_bytes(b"server")
    return node, server


def test_retry_reuses_current_runtime_without_recopying_legacy_config(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    durable = tmp_path / "durable"
    legacy.mkdir(mode=0o700)
    durable.mkdir(mode=0o700)
    (legacy / "config.json").write_text('{"source":"legacy"}', encoding="utf-8")
    current = b'{"source":"current","userChange":true}'
    (durable / "config.json").write_bytes(current)
    node, server = _runtime_files(tmp_path)
    calls: list[list[str]] = []

    def run(arguments, **_options):
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    paths = MigrationPaths(durable_root=durable, cache_root=tmp_path / "cache")
    with ApplicationMigration(paths) as authority:
        authority.begin()
        authority.publish_standard_data(
            legacy_root=legacy,
            node_executable=node,
            server_js=server,
            run=run,
            reuse_existing=True,
        )

    assert calls == [[str(node), str(server), "--initialize-application-data"]]
    assert (durable / "config.json").read_bytes() == current
    journal = json.loads(paths.journal_path.read_text(encoding="utf-8"))
    assert journal["phase"] == "data_published"
    assert journal["dataManifest"]["config.json"]["reused"] is True


def test_retained_retry_requires_untampered_recovery_evidence(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    durable = tmp_path / "durable"
    legacy.mkdir(mode=0o700)
    durable.mkdir(mode=0o700)
    (legacy / "config.json").write_text('{"source":"legacy"}', encoding="utf-8")
    paths = MigrationPaths(durable_root=durable, cache_root=tmp_path / "cache")

    with ApplicationMigration(paths) as authority:
        first = authority.begin()
        preflight = preflight_standard_outputs(legacy, durable)
        (durable / "config.json").write_text('{"source":"migrated"}', encoding="utf-8")
        root = os.fstat(authority._durable_root_fd)
        authority._journal = {
            **authority._journal,
            "preflightDataManifest": preflight,
            "durableDirectory": {"device": root.st_dev, "inode": root.st_ino},
            "runtimeInitializationStarted": True,
            "serviceNonce": "a" * 32,
        }
        authority.record_transaction_output_identities(preflight)
        authority.transition("rolling_back", intent={"action": "restore-legacy"})
        authority.retain_started_transaction_outputs()
        authority.transition("rolled_back", intent={"action": "rollback-complete"})

    (durable / "config.json").write_text('{"source":"current"}', encoding="utf-8")
    journal = json.loads(paths.journal_path.read_text(encoding="utf-8"))
    assert _retained_runtime_retry_transaction(paths, journal) == first["transactionId"]

    retained = (
        paths.journal_dir
        / "retained-runtime-data"
        / first["transactionId"]
        / "config.json"
    )
    retained.write_text("tampered", encoding="utf-8")
    with pytest.raises(MigrationBlocked, match="retained runtime recovery changed"):
        _retained_runtime_retry_transaction(paths, journal)
