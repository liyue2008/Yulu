#!/usr/bin/env python3
"""Install and inspect Yulu's optional local streaming-caption runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if __name__ == "__main__":
    # Isolated Python omits the script directory; trust only our bundled siblings.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from application_paths import DURABLE_DATA_DIR, MODELS_DIR

SHERPA_VERSION = "1.13.2"


class ModelAsset:
    """A downloadable model asset managed by the local caption runtime."""

    def __init__(
        self,
        *,
        key: str,
        name: str,
        url: str,
        sha256: str,
        files: tuple[str, ...],
        file_sha256: dict[str, str],
        kind: str,
        archive_bytes: int,
        installed_bytes: int,
    ) -> None:
        self.key = key
        self.name = name
        self.url = url
        self.sha256 = sha256
        self.files = files
        self.file_sha256 = file_sha256
        self.kind = kind  # "tarbz2" (archive with top-level dir == name) or "file" (bare onnx)
        self.archive_bytes = archive_bytes
        self.installed_bytes = installed_bytes


MODELS: dict[str, ModelAsset] = {
    "streaming": ModelAsset(
        key="streaming",
        name="sherpa-onnx-streaming-paraformer-bilingual-zh-en",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2"
        ),
        sha256="5462a1fce42693deae572af1e8c4687124b12aa85fe61ff4d3168bb5280e205f",
        files=("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"),
        file_sha256={
            "tokens.txt": "59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6",
            "encoder.int8.onnx": "81a70226a8934e6ed92aa1d4fc486b428b5398e2f2619ed4897b7294cab90e9a",
            "decoder.int8.onnx": "f3cca9f77bb9d93c8fcbfb63ae617b6b1ee96818df3aa3b151c40658fe38594f",
        },
        kind="tarbz2",
        archive_bytes=252_000_000,
        installed_bytes=238_000_000,
    ),
    "offline-final": ModelAsset(
        key="offline-final",
        name="sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2"
        ),
        sha256="1b158e9d46715ed1cd387402b125de26f2e09bf2cb73926414b7fbd74d1973e2",
        files=("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"),
        file_sha256={
            "tokens.txt": "6907215aeb034f6926b26bf8abfd650f756781622480a2342ec1f29b2072cafe",
            "encoder.int8.onnx": "e60cfef737a0ea324846a64eca8b9dae35898f353f4e34b62ad7e536e2d86add",
            "decoder.int8.onnx": "c08b9d0297ed17ad84087085e27a4adedcc4e8b3ef14770369f1665681cc507d",
        },
        kind="tarbz2",
        archive_bytes=1_469_513_701,
        installed_bytes=1_738_971_645,
    ),
    "silero-vad": ModelAsset(
        key="silero-vad",
        name="silero-vad",
        url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        files=("silero_vad.onnx",),
        file_sha256={
            "silero_vad.onnx": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        },
        kind="file",
        archive_bytes=643_854,
        installed_bytes=643_854,
    ),
}

MODEL_NAME = MODELS["streaming"].name  # legacy alias: the streaming model directory
MODEL_URL = MODELS["streaming"].url
MODEL_SHA256 = MODELS["streaming"].sha256
MODEL_FILES = MODELS["streaming"].files
MODEL_FILE_SHA256 = MODELS["streaming"].file_sha256
DEFAULT_INSTALL_MODELS = ("streaming", "silero-vad")
PACK_DEFINITION_PATH = Path(__file__).with_name("local_caption_runtime_pack.json")


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def runtime_paths(config_dir: Path, *, models_dir: Path | None = None) -> dict[str, Path]:
    runtime = config_dir / "local-caption"
    pack = runtime / "YuluLocalCaptionRuntime.bundle"
    model_root = models_dir or config_dir / "models"
    paths: dict[str, Path] = {
        "runtime": runtime,
        "pack": pack,
        "site_packages": pack / "Contents" / "Resources" / "site-packages",
        "python": Path(os.environ.get("YULU_PYTHON", sys.executable)).resolve(),
        "manifest": runtime / "manifest.json",
        "model_root": model_root,
    }
    for asset in MODELS.values():
        paths[f"model_{asset.key}"] = model_root / asset.name
    paths["model"] = paths["model_streaming"]
    return paths


def _model_complete(model_dir: Path, asset: ModelAsset | None = None) -> bool:
    asset = asset or MODELS["streaming"]
    return all((model_dir / name).is_file() and (model_dir / name).stat().st_size > 0 for name in asset.files)


def _verify_model_hashes(model_dir: Path, asset: ModelAsset | None = None) -> bool:
    asset = asset or MODELS["streaming"]
    if not _model_complete(model_dir, asset):
        return False
    for name, expected in asset.file_sha256.items():
        digest = hashlib.sha256()
        with (model_dir / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            return False
    return True


def _load_runtime_pack_definition() -> dict[str, Any]:
    try:
        definition = json.loads(PACK_DEFINITION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("local caption Runtime Pack definition is unavailable") from exc
    required = {
        "schema": 1,
        "id": "sherpa-onnx-1.13.2-cp313-macos-arm64",
        "version": SHERPA_VERSION,
        "architecture": "arm64",
        "pythonAbi": "cp313",
        "bundleName": "YuluLocalCaptionRuntime.bundle",
        "bundleIdentifier": "com.yulu.runtime.local-caption",
        "teamIdentifier": "WMU9678ZQL",
    }
    for key, expected in required.items():
        if definition.get(key) != expected:
            raise RuntimeError(f"invalid local caption Runtime Pack definition: {key}")
    if "{tag}" not in str(definition.get("assetUrlTemplate", "")):
        raise RuntimeError("invalid local caption Runtime Pack asset URL")
    return definition


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pack_manifest(pack: Path, definition: dict[str, Any]) -> dict[str, Any]:
    manifest_path = pack / "Contents" / "Resources" / "runtime-pack.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Runtime Pack manifest is missing or invalid") from exc
    for key in ("schema", "id", "version", "architecture", "pythonAbi"):
        if manifest.get(key) != definition.get(key):
            raise RuntimeError(f"Runtime Pack manifest identity mismatch: {key}")
    return manifest


def _verify_pack_payload(pack: Path, definition: dict[str, Any]) -> None:
    site_packages = pack / "Contents" / "Resources" / "site-packages"
    if not site_packages.is_dir():
        raise RuntimeError("Runtime Pack site-packages is missing")
    manifest = _pack_manifest(pack, definition)
    expected: dict[str, str] = {}
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict):
            raise RuntimeError("Runtime Pack inventory entry is invalid")
        relative = entry.get("path")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or relative in expected
        ):
            raise RuntimeError("Runtime Pack inventory entry is invalid")
        expected[relative] = digest
    actual: dict[str, Path] = {}
    for path in site_packages.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Runtime Pack payload must not contain symlinks")
        if path.is_file():
            actual[path.relative_to(site_packages).as_posix()] = path
    if set(actual) != set(expected):
        raise RuntimeError("Runtime Pack file inventory does not match its payload")
    for relative, path in actual.items():
        if _sha256_file(path) != expected[relative]:
            raise RuntimeError(f"Runtime Pack payload hash mismatch: {relative}")


def _codesign_metadata(path: Path, *, arm64_only: bool = False) -> tuple[str, str]:
    result = subprocess.run(
        ["/usr/bin/codesign", "--display", "--verbose=2", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Runtime Pack signature metadata unavailable: {path.name}")
    output = result.stdout + result.stderr
    identifier = ""
    team = ""
    code_format = ""
    for line in output.splitlines():
        if line.startswith("Identifier="):
            identifier = line.split("=", 1)[1].strip()
        elif line.startswith("TeamIdentifier="):
            team = line.split("=", 1)[1].strip()
        elif line.startswith("Format="):
            code_format = line.split("=", 1)[1].strip()
    # codesign is part of macOS; lipo is an Xcode/CLT shim on a clean machine.
    # Its Format lists the complete architecture set, including fat binaries.
    if arm64_only and code_format not in {
        "Mach-O thin (arm64)", "Mach-O universal (arm64)",
    }:
        raise RuntimeError(f"Runtime Pack native code is not arm64-only: {path.name}")
    return identifier, team


def _verify_pack_code_signatures(pack: Path, definition: dict[str, Any]) -> None:
    verified = subprocess.run(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(pack)],
        capture_output=True,
        text=True,
    )
    if verified.returncode != 0:
        raise RuntimeError("Runtime Pack signature verification failed")
    identifier, pack_team = _codesign_metadata(pack)
    if identifier != definition["bundleIdentifier"]:
        raise RuntimeError("Runtime Pack bundle identifier is invalid")
    _, python_team = _codesign_metadata(Path(sys.executable).resolve())
    allow_adhoc = os.environ.get("YULU_ALLOW_ADHOC_RUNTIME_PACK") == "1"
    development_smoke = os.environ.get("YULU_DEV_SMOKE") == "1"
    expected_team = str(definition["teamIdentifier"])
    pack_has_team = bool(pack_team and pack_team != "not set")
    python_has_team = bool(python_team and python_team != "not set")
    if pack_has_team:
        if pack_team != expected_team:
            raise RuntimeError("Runtime Pack is not signed by the Application Runtime Team")
        if python_has_team:
            if python_team != pack_team:
                raise RuntimeError("Runtime Pack is not signed by the Application Runtime Team")
        elif not development_smoke:
            raise RuntimeError("Runtime Pack and bundled Python require the same Developer ID Team")
    elif python_has_team or not allow_adhoc:
        raise RuntimeError("Runtime Pack and bundled Python require the same Developer ID Team")

    site_packages = pack / "Contents" / "Resources" / "site-packages"
    for path in site_packages.rglob("*"):
        if not path.is_file():
            continue
        description = subprocess.run(
            ["/usr/bin/file", "-b", str(path)], capture_output=True, text=True
        )
        if description.returncode != 0 or "Mach-O" not in description.stdout:
            continue
        signature = subprocess.run(
            ["/usr/bin/codesign", "--verify", "--strict", str(path)],
            capture_output=True,
            text=True,
        )
        if signature.returncode != 0:
            raise RuntimeError(f"Runtime Pack native signature is invalid: {path.name}")
        _, native_team = _codesign_metadata(path, arm64_only=True)
        if pack_team and pack_team != "not set" and native_team != pack_team:
            raise RuntimeError(f"Runtime Pack native code has the wrong signing Team: {path.name}")


def _runtime_pack_ok(pack: Path, definition: dict[str, Any]) -> bool:
    if not pack.is_dir():
        return False
    try:
        _verify_pack_payload(pack, definition)
        _verify_pack_code_signatures(pack, definition)
        return True
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return False


def verify_runtime_pack(pack: Path) -> None:
    definition = _load_runtime_pack_definition()
    expected = definition["bundleName"]
    if pack.name != expected:
        raise RuntimeError(f"unexpected Runtime Pack bundle name: {pack.name}")
    _verify_pack_payload(pack, definition)
    _verify_pack_code_signatures(pack, definition)


def _sherpa_import_ok(python: Path, site_packages: Path) -> bool:
    if not python.is_file():
        return False
    try:
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-S",
                "-B",  # -I ignores PYTHONDONTWRITEBYTECODE; keep signed code immutable.
                "-c",
                (
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "import sherpa_onnx; print(sherpa_onnx.__version__)"
                ),
                str(site_packages),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
        return result.returncode == 0 and result.stdout.strip() == SHERPA_VERSION
    except (OSError, subprocess.SubprocessError):
        return False


def status(config_dir: Path, *, models_dir: Path | None = None) -> dict[str, Any]:
    paths = runtime_paths(config_dir, models_dir=models_dir)
    try:
        definition = _load_runtime_pack_definition()
        pack_ok = _runtime_pack_ok(paths["pack"], definition)
    except (OSError, RuntimeError, ValueError):
        pack_ok = False
    runtime_ok = pack_ok and _sherpa_import_ok(paths["python"], paths["site_packages"])
    model_ok = _verify_model_hashes(paths["model"])
    models: dict[str, dict[str, Any]] = {}
    for asset in MODELS.values():
        directory = paths[f"model_{asset.key}"]
        models[asset.key] = {
            "ready": _verify_model_hashes(directory, asset),
            "bytes": _dir_size(directory),
            "name": asset.name,
            "provider": (
                "sherpa-onnx-fire-red-asr-large-int8"
                if asset.key == "offline-final"
                else "sherpa-onnx-paraformer-int8" if asset.key == "streaming" else "silero-vad"
            ),
            "archiveBytes": asset.archive_bytes,
            "installedBytes": asset.installed_bytes,
        }
    return {
        "installed": runtime_ok and model_ok,
        "runtimeReady": runtime_ok,
        "modelReady": model_ok,
        "provider": "sherpa-onnx-paraformer-int8",
        "version": SHERPA_VERSION,
        "model": MODEL_NAME,
        "runtimeBytes": _dir_size(paths["runtime"]),
        "modelBytes": _dir_size(paths["model"]),
        "python": str(paths["python"]),
        "pythonPath": str(paths["site_packages"]),
        "runtimePack": str(paths["pack"]),
        "modelDir": str(paths["model"]),
        "offlineFinalReady": models["offline-final"]["ready"],
        "offlineFinalBytes": models["offline-final"]["bytes"],
        "vadReady": models["silero-vad"]["ready"],
        "models": models,
    }


def _release_tag() -> str:
    override = os.environ.get("YULU_LOCAL_CAPTION_RUNTIME_PACK_TAG", "").strip()
    if override:
        return override
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.suffix != ".app":
            continue
        info = parent / "Contents" / "Info.plist"
        try:
            with info.open("rb") as source:
                version = str(plistlib.load(source).get("YuluVersion", "")).strip()
        except (OSError, plistlib.InvalidFileException):
            break
        if version:
            return version if version.startswith("v") else f"v{version}"
    raise RuntimeError("无法确定当前 Yulu 版本，不能选择 Runtime Pack")


def _safe_extract_runtime_pack(archive: Path, destination: Path, bundle_name: str) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        seen: set[str] = set()
        for entry in bundle.infolist():
            relative = Path(entry.filename)
            if (
                not entry.filename
                or entry.filename.startswith("/")
                or ".." in relative.parts
                or relative.parts[0] != bundle_name
                or entry.filename in seen
            ):
                raise RuntimeError("Runtime Pack archive contains an unsafe path")
            seen.add(entry.filename)
            mode = (entry.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise RuntimeError("Runtime Pack archive must not contain symlinks")
            target = destination.joinpath(*relative.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            permissions = stat.S_IMODE(mode)
            if permissions:
                target.chmod(permissions)
    pack = destination / bundle_name
    if not pack.is_dir():
        raise RuntimeError("Runtime Pack archive is missing its signed bundle")
    return pack


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        return


def _install_runtime_pack(runtime_dir: Path, definition: dict[str, Any]) -> None:
    target = runtime_dir / str(definition["bundleName"])
    if _runtime_pack_ok(target, definition):
        _emit("progress", phase="runtime", message="本地识别 Runtime Pack 已就绪")
        return
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".local-caption-pack-", dir=runtime_dir.parent) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "runtime-pack.zip"
        supplied = os.environ.get("YULU_LOCAL_CAPTION_RUNTIME_PACK_ARCHIVE", "").strip()
        if supplied:
            source = Path(supplied)
            if not source.is_file():
                raise RuntimeError("指定的 Runtime Pack 不存在")
            shutil.copy2(source, archive)
        else:
            tag = _release_tag()
            url = str(definition["assetUrlTemplate"]).format(tag=tag)
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.hostname != "github.com"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or not parsed.path.startswith("/Nowhitestar/Yulu/releases/download/")
            ):
                raise RuntimeError("Runtime Pack download URL is invalid")
            _emit("progress", phase="runtime", message=f"下载本地识别 Runtime Pack {tag}")
            downloaded = subprocess.run(
                [
                    "/usr/bin/curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--proto",
                    "=https",
                    "--proto-redir",
                    "=https",
                    "--output",
                    str(archive),
                    url,
                ],
                capture_output=True,
                text=True,
            )
            if downloaded.returncode != 0:
                raise RuntimeError("Runtime Pack download failed")
        staging = temporary_path / "staging"
        staging.mkdir()
        pack = _safe_extract_runtime_pack(archive, staging, str(definition["bundleName"]))
        _verify_pack_payload(pack, definition)
        _verify_pack_code_signatures(pack, definition)

        runtime_dir.mkdir(parents=True, exist_ok=True)
        backup = runtime_dir / f".{definition['bundleName']}.previous"
        _remove_tree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(pack, target)
            _verify_pack_payload(target, definition)
            _verify_pack_code_signatures(target, definition)
        except Exception:
            _remove_tree(target)
            if backup.exists():
                os.replace(backup, target)
            raise
        _remove_tree(backup)


def _copy_benchmark_model(model_dir: Path, asset: ModelAsset) -> bool:
    source = Path.home() / ".cache/yulu-asr-benchmark/models" / asset.name
    if not _verify_model_hashes(source, asset):
        return False
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in asset.files:
        shutil.copy2(source / name, model_dir / name)
    return True


def _ensure_disk_space(asset: ModelAsset, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(destination_root)
    needed = asset.archive_bytes + asset.installed_bytes
    if usage.free < needed:
        raise RuntimeError(
            f"磁盘空间不足：安装 {asset.name} 约需 {needed / 1_000_000_000:.1f} GB（含下载缓存），"
            f"当前可用 {usage.free / 1_000_000_000:.1f} GB"
        )


def _download_asset(asset: ModelAsset, model_dir: Path) -> None:
    _ensure_disk_space(asset, model_dir.parent)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yulu-caption-model-") as temp:
        archive = Path(temp) / (f"{asset.name}.tar.bz2" if asset.kind == "tarbz2" else asset.files[0])
        last_percent = -1

        def progress(blocks: int, block_size: int, total: int) -> None:
            nonlocal last_percent
            if total <= 0:
                return
            percent = min(100, blocks * block_size * 100 // total)
            if percent >= last_percent + 5:
                last_percent = percent
                _emit("progress", phase="download", percent=percent, message=f"下载模型 {percent}%")

        urllib.request.urlretrieve(  # noqa: S310 -- fixed HTTPS model URL
            asset.url,
            archive,
            progress,
        )
        digest_state = hashlib.sha256()
        with archive.open("rb") as archive_source:
            for chunk in iter(lambda: archive_source.read(1024 * 1024), b""):
                digest_state.update(chunk)
        digest = digest_state.hexdigest()
        if digest != asset.sha256:
            raise RuntimeError(f"模型校验失败: expected {asset.sha256}, got {digest}")
        staging = Path(temp) / "model"
        staging.mkdir()
        if asset.kind == "file":
            shutil.copy2(archive, staging / asset.files[0])
        else:
            with tarfile.open(archive, "r:bz2") as bundle:
                members = {member.name: member for member in bundle.getmembers()}
                for name in asset.files:
                    member_name = f"{asset.name}/{name}"
                    member = members.get(member_name)
                    if member is None or not member.isfile():
                        raise RuntimeError(f"模型包缺少 {member_name}")
                    member_source = bundle.extractfile(member)
                    if member_source is None:
                        raise RuntimeError(f"无法读取 {member_name}")
                    with member_source, (staging / name).open("wb") as target:
                        shutil.copyfileobj(member_source, target)
        if not _verify_model_hashes(staging, asset):
            raise RuntimeError("解压后的模型校验失败")
        # Safe replace: move the previous install aside first, publish the new
        # model, and only delete the old tree after the new one is verified in
        # place. Any failure restores the previous install instead of leaving
        # the target deleted-but-not-replaced.
        backup = model_dir.with_name(f".{model_dir.name}.previous")
        _remove_tree(backup)
        had_previous = model_dir.exists()
        if had_previous:
            os.replace(model_dir, backup)

        def _restore_previous() -> None:
            _remove_tree(model_dir)
            if had_previous and backup.exists():
                os.replace(backup, model_dir)

        try:
            try:
                shutil.move(str(staging), str(model_dir))
            except OSError as exc:
                raise RuntimeError("本地模型发布失败") from exc
            if not _verify_model_hashes(model_dir, asset):
                raise RuntimeError("模型发布后校验失败")
        except Exception:
            _restore_previous()
            raise
        _remove_tree(backup)


def install(
    config_dir: Path,
    *,
    models_dir: Path | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    paths = runtime_paths(config_dir, models_dir=models_dir)
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    definition = _load_runtime_pack_definition()
    _install_runtime_pack(paths["runtime"], definition)
    selected = list(DEFAULT_INSTALL_MODELS) if model is None else [model]
    for key in selected:
        asset = MODELS[key]
        target = paths[f"model_{asset.key}"]
        if _verify_model_hashes(target, asset):
            continue
        _emit("progress", phase="model", message=f"准备 {asset.name}")
        if asset.key == "streaming" and _copy_benchmark_model(target, asset):
            continue
        _download_asset(asset, target)
    manifest = {
        "schema": 1,
        "provider": "sherpa-onnx-paraformer-int8",
        "version": SHERPA_VERSION,
        "model": MODEL_NAME,
        "modelSha256": MODEL_SHA256,
        "runtimePack": definition["id"],
        "models": {
            asset.key: {"name": asset.name, "sha256": asset.sha256}
            for asset in MODELS.values()
            if _verify_model_hashes(paths[f"model_{asset.key}"], asset)
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = status(config_dir, models_dir=models_dir)
    for key in selected:
        if not result["models"][key]["ready"]:
            raise RuntimeError(f"模型 {MODELS[key].name} 安装后未通过自检")
    if "streaming" in selected and not result["installed"]:
        raise RuntimeError("本地实时转录运行时安装后未通过自检")
    return result


def uninstall(
    config_dir: Path,
    *,
    models_dir: Path | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    paths = runtime_paths(config_dir, models_dir=models_dir)
    if model is None:
        _remove_tree(paths["runtime"])
        for asset in MODELS.values():
            _remove_tree(paths[f"model_{asset.key}"])
    else:
        _remove_tree(paths[f"model_{model}"])
    return status(config_dir, models_dir=models_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage Yulu local caption runtime")
    parser.add_argument("action", choices=("status", "install", "uninstall"))
    parser.add_argument("--config-dir", type=Path, default=DURABLE_DATA_DIR)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--model", choices=tuple(MODELS), help="限定操作单个模型资产；缺省时安装/卸载全部默认资产")
    args = parser.parse_args(argv)
    config_dir = args.config_dir.expanduser()
    models_dir = args.models_dir.expanduser() if args.models_dir else MODELS_DIR
    try:
        if args.action == "install":
            result = install(config_dir, models_dir=models_dir, model=args.model)
        elif args.action == "uninstall":
            result = uninstall(config_dir, models_dir=models_dir, model=args.model)
        else:
            result = status(config_dir, models_dir=models_dir)
        _emit("result", ok=True, status=result)
        return 0
    except Exception as exc:
        _emit("result", ok=False, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
