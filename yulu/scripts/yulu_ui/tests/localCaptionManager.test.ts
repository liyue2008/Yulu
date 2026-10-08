import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LocalCaptionManager } from "../src/localCaptionManager.js";

const roots: string[] = [];
const originalYuluPython = process.env.YULU_PYTHON;
const originalLocalCaptionModelDir = process.env.YULU_LOCAL_CAPTION_MODEL_DIR;
const originalPythonPath = process.env.PYTHONPATH;
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
  if (originalYuluPython === undefined) delete process.env.YULU_PYTHON;
  else process.env.YULU_PYTHON = originalYuluPython;
  if (originalLocalCaptionModelDir === undefined) delete process.env.YULU_LOCAL_CAPTION_MODEL_DIR;
  else process.env.YULU_LOCAL_CAPTION_MODEL_DIR = originalLocalCaptionModelDir;
  if (originalPythonPath === undefined) delete process.env.PYTHONPATH;
  else process.env.PYTHONPATH = originalPythonPath;
});

function fixture(
  selected: boolean | (() => boolean) = true,
) {
  const root = mkdtempSync(join(tmpdir(), "yulu-caption-manager-"));
  roots.push(root);
  const scriptDir = join(root, "scripts");
  const configDir = join(root, "config");
  const modelDir = join(configDir, "models/sherpa-onnx-streaming-paraformer-bilingual-zh-en");
  const packSitePackages = join(
    configDir,
    "local-caption/YuluLocalCaptionRuntime.bundle/Contents/Resources/site-packages",
  );
  mkdirSync(scriptDir, { recursive: true });
  mkdirSync(modelDir, { recursive: true });
  mkdirSync(packSitePackages, { recursive: true });
  const python = existsSync("/opt/homebrew/bin/python3")
    ? "/opt/homebrew/bin/python3"
    : execFileSync("which", ["python3"], { encoding: "utf8" }).trim();
  process.env.YULU_PYTHON = python;
  for (const name of ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]) {
    writeFileSync(join(modelDir, name), name);
  }
  const workerPath = join(scriptDir, "sherpa_caption_worker.py");
  writeFileSync(workerPath, [
    "#!/usr/bin/env python3",
    "import json, sys",
    "for line in sys.stdin:",
    "  req=json.loads(line); action=req['action']",
    "  result={'ready': True} if action == 'ping' else ({'updates': {}} if action in ('finish','feed') else {})",
    "  print(json.dumps({'id': req['id'], 'ok': True, 'result': result}), flush=True)",
    "  if action == 'shutdown': break",
  ].join("\n"));
  chmodSync(workerPath, 0o755);
  return new LocalCaptionManager({
    scriptDir,
    configDir,
    selected: typeof selected === "function" ? selected : () => selected,
  });
}

describe("LocalCaptionManager", () => {
  it("uses the standard Models directory supplied by the Host", () => {
    const root = mkdtempSync(join(tmpdir(), "yulu-caption-standard-models-"));
    roots.push(root);
    const scriptDir = join(root, "scripts");
    const configDir = join(root, "Library/Application Support/Yulu");
    const modelsDir = join(configDir, "Standard Models");
    const modelDir = join(modelsDir, "sherpa-onnx-streaming-paraformer-bilingual-zh-en");
    const packSitePackages = join(
      configDir,
      "local-caption/YuluLocalCaptionRuntime.bundle/Contents/Resources/site-packages",
    );
    mkdirSync(scriptDir, { recursive: true });
    mkdirSync(modelDir, { recursive: true });
    mkdirSync(packSitePackages, { recursive: true });
    writeFileSync(join(scriptDir, "sherpa_caption_worker.py"), "");
    for (const name of ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]) {
      writeFileSync(join(modelDir, name), name);
    }
    process.env.YULU_PYTHON = process.execPath;
    delete process.env.YULU_LOCAL_CAPTION_MODEL_DIR;

    const manager = new LocalCaptionManager({
      scriptDir,
      configDir,
      modelsDir,
      selected: () => true,
    });

    expect(manager.status()).toMatchObject({ installed: true });
    expect(manager.status().modelBytes).toBeGreaterThan(0);
  });

  it("reports a complete runtime and runs a warm self-test", async () => {
    const manager = fixture();
    expect(manager.status()).toMatchObject({
      installed: true,
      ready: true,
      operation: "idle",
      offlineModelReady: false,
      offlineModelBytes: 0,
      vadReady: false,
    });

    await expect(manager.test()).resolves.toMatchObject({ ok: true, provider: "sherpa-onnx-paraformer-int8" });
    expect(manager.status().message).toMatch(/测试通过/);
    await manager.close();
  });

  it("does not start the local model when xAI is selected", async () => {
    const manager = fixture(false);
    await expect(manager.start("zh")).rejects.toThrow("未选择本地音频引擎");
    await expect(manager.test()).resolves.toMatchObject({ ok: true });
    expect(manager.status().sessionActive).toBe(false);
    await manager.close();
  });

  it("fails visibly instead of switching engines for unsupported Japanese", async () => {
    const manager = fixture();
    await expect(manager.start("ja")).rejects.toThrow("仅支持中英文");
    expect(manager.status().sessionActive).toBe(false);
    await manager.close();
  });

  it("can install while xAI is selected without warming the unselected local engine", async () => {
    const manager = fixture(false);
    vi.spyOn(
      manager as unknown as { runInstaller(action: "install" | "uninstall"): Promise<void> },
      "runInstaller",
    ).mockResolvedValue();

    await expect(manager.install()).resolves.toMatchObject({
      installed: true,
      operation: "idle",
    });
    await manager.close();
  });

  it("passes the Host standard Models directory to the installer", async () => {
    const root = mkdtempSync(join(tmpdir(), "yulu-caption-installer-models-"));
    roots.push(root);
    const scriptDir = join(root, "scripts");
    const configDir = join(root, "Library/Application Support/Yulu");
    const modelsDir = join(configDir, "Standard Models");
    const argvPath = join(root, "installer-argv.json");
    mkdirSync(scriptDir, { recursive: true });
    writeFileSync(join(scriptDir, "local_caption_runtime.py"), [
      "import json, sys",
      `open(${JSON.stringify(argvPath)}, 'w').write(json.dumps(sys.argv[1:]))`,
      "",
    ].join("\n"));
    process.env.YULU_PYTHON = existsSync("/opt/homebrew/bin/python3")
      ? "/opt/homebrew/bin/python3"
      : execFileSync("which", ["python3"], { encoding: "utf8" }).trim();

    const manager = new LocalCaptionManager({
      scriptDir,
      configDir,
      modelsDir,
      selected: () => false,
    });
    await manager.install();

    expect(JSON.parse(readFileSync(argvPath, "utf8"))).toEqual([
      "install",
      "--config-dir",
      configDir,
      "--models-dir",
      modelsDir,
    ]);
  });

  it("starts the installer isolated from inherited Python startup code and modules", async () => {
    const manager = fixture(false);
    const root = roots.at(-1)!;
    const malicious = join(root, "malicious-python-path");
    const startupMarker = join(root, "sitecustomize-ran");
    const shadowMarker = join(root, "shadow-module-ran");
    const environmentMarker = join(root, "python-environment.txt");
    mkdirSync(malicious, { recursive: true });
    writeFileSync(join(malicious, "sitecustomize.py"), [
      "from pathlib import Path",
      `Path(${JSON.stringify(startupMarker)}).write_text('unsafe')`,
      "",
    ].join("\n"));
    writeFileSync(join(malicious, "fractions.py"), [
      "from pathlib import Path",
      `Path(${JSON.stringify(shadowMarker)}).write_text('unsafe')`,
      "",
    ].join("\n"));
    writeFileSync(join(root, "scripts/local_caption_runtime.py"), [
      "from pathlib import Path",
      "import fractions, os",
      `Path(${JSON.stringify(environmentMarker)}).write_text(','.join(sorted(key for key in os.environ if key.startswith('PYTHON'))))`,
      "",
    ].join("\n"));
    process.env.PYTHONPATH = malicious;

    await expect(manager.install()).resolves.toMatchObject({ operation: "idle" });
    expect(existsSync(startupMarker)).toBe(false);
    expect(existsSync(shadowMarker)).toBe(false);
    expect(readFileSync(environmentMarker, "utf8")).toBe("");
    await manager.close();
  });

  it("releases the resident worker when xAI is selected while idle", async () => {
    let selected = true;
    const manager = fixture(() => selected);
    await manager.test();
    expect((manager as unknown as { engine: unknown }).engine).not.toBeNull();

    selected = false;
    await manager.syncSelection();

    expect((manager as unknown as { engine: unknown }).engine).toBeNull();
    await manager.close();
  });

  it("reports a worker warmup failure instead of claiming the model is ready", async () => {
    const manager = fixture();
    const slot = manager as unknown as { engine: { warm(): Promise<void>; close(): Promise<void> } | null };
    slot.engine = {
      warm: vi.fn().mockRejectedValue(new Error("model load failed")),
      close: vi.fn().mockResolvedValue(undefined),
    };

    await expect(manager.warm()).rejects.toThrow("model load failed");
    expect(manager.status()).toMatchObject({ ready: false, error: "model load failed" });
    await manager.close();
  });

  it("forwards the selected model key to install and uninstall", async () => {
    const root = mkdtempSync(join(tmpdir(), "yulu-caption-model-flag-"));
    roots.push(root);
    const scriptDir = join(root, "scripts");
    const configDir = join(root, "Library/Application Support/Yulu");
    const modelsDir = join(configDir, "Standard Models");
    const argvPath = join(root, "installer-argv.json");
    mkdirSync(scriptDir, { recursive: true });
    writeFileSync(join(scriptDir, "local_caption_runtime.py"), [
      "import json, sys",
      `open(${JSON.stringify(argvPath)}, 'w').write(json.dumps(sys.argv[1:]))`,
      "",
    ].join("\n"));
    process.env.YULU_PYTHON = existsSync("/opt/homebrew/bin/python3")
      ? "/opt/homebrew/bin/python3"
      : execFileSync("which", ["python3"], { encoding: "utf8" }).trim();

    const manager = new LocalCaptionManager({
      scriptDir,
      configDir,
      modelsDir,
      selected: () => false,
    });

    await manager.install({ model: "offline-final" });
    expect(JSON.parse(readFileSync(argvPath, "utf8"))).toEqual([
      "install",
      "--config-dir",
      configDir,
      "--models-dir",
      modelsDir,
      "--model",
      "offline-final",
    ]);

    await manager.uninstall({ model: "silero-vad" });
    expect(JSON.parse(readFileSync(argvPath, "utf8"))).toEqual([
      "uninstall",
      "--config-dir",
      configDir,
      "--models-dir",
      modelsDir,
      "--model",
      "silero-vad",
    ]);
  });

  it("refuses the offline smoke test before the offline assets are installed", async () => {
    const manager = fixture();
    await expect(manager.testOffline()).rejects.toThrow("离线高质量转录模型尚未安装");
    await manager.close();
  });

  it("runs the offline smoke test through the one-shot worker when assets exist", async () => {
    const manager = fixture();
    const root = roots.at(-1)!;
    const offlineDir = join(root, "config/models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16");
    const vadDir = join(root, "config/models/silero-vad");
    mkdirSync(offlineDir, { recursive: true });
    mkdirSync(vadDir, { recursive: true });
    for (const name of ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]) {
      writeFileSync(join(offlineDir, name), name);
    }
    writeFileSync(join(vadDir, "silero_vad.onnx"), "vad");

    const runOfflineWorker = vi.fn().mockResolvedValue(0);
    vi.spyOn(
      manager as unknown as { runOfflineWorker(...args: unknown[]): Promise<number> },
      "runOfflineWorker",
    ).mockImplementation(runOfflineWorker);

    await expect(manager.testOffline()).resolves.toEqual({
      ok: true,
      provider: "sherpa-onnx-fire-red-asr-large-int8",
      loadMs: expect.any(Number),
      segments: 0,
    });
    expect(runOfflineWorker).toHaveBeenCalledTimes(1);
    expect(manager.status()).toMatchObject({ offlineModelReady: true, vadReady: true });
    await manager.close();
  });
});
