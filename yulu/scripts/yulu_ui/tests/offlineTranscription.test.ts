import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { cpus } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentUnavailableError } from "../src/agentGateway.js";
import {
  OfflineTranscriptionService,
  offlineThreadCount,
} from "../src/offlineTranscription.js";

const roots: string[] = [];
const originalYuluPython = process.env.YULU_PYTHON;
const originalStubArgs = process.env.STUB_ARGS_PATH;

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
  if (originalYuluPython === undefined) delete process.env.YULU_PYTHON;
  else process.env.YULU_PYTHON = originalYuluPython;
  if (originalStubArgs === undefined) delete process.env.STUB_ARGS_PATH;
  else process.env.STUB_ARGS_PATH = originalStubArgs;
  vi.restoreAllMocks();
});

function pythonBin(): string {
  const direct = "/opt/homebrew/bin/python3";
  if (existsSync(direct)) return direct;
  return execFileSync("which", ["python3"], { encoding: "utf8" }).trim();
}

/** Python stub standing in for sherpa_offline_worker.py: records argv and prints protocol events. */
const STUB_WORKER = [
  "#!/usr/bin/env python3",
  "import json, os, sys",
  "args = sys.argv[1:]",
  "with open(os.environ['STUB_ARGS_PATH'], 'w') as sink:",
  "    sink.write(json.dumps(args))",
  "print(json.dumps({'event': 'progress', 'phase': 'vad', 'source': 'mic', 'segments': 2, 'message': 'mic: ok'}), flush=True)",
  "print(json.dumps({'event': 'progress', 'phase': 'decode', 'percent': 100, 'message': 'done'}), flush=True)",
  "print(json.dumps({",
  "    'event': 'result',",
  "    'provider': 'sherpa-onnx-fire-red-asr-large-int8',",
  "    'sources': {",
  "        'mic': [",
  "            {'startMs': 0, 'endMs': 1000, 'text': '会议开始'},",
  "            {'startMs': 2000, 'endMs': 3000, 'text': '讨论预算'},",
  "        ],",
  "        'system': [",
  "            {'startMs': 100, 'endMs': 1100, 'text': '会议开始'},",
  "            {'startMs': 4000, 'endMs': 5000, 'text': '系统提示'},",
  "        ],",
  "    },",
  "}), flush=True)",
  "",
].join("\n");

const HANG_WORKER = [
  "#!/usr/bin/env python3",
  "import time",
  "time.sleep(600)",
  "",
].join("\n");

interface Fixture {
  root: string;
  scriptDir: string;
  configDir: string;
  offlineModelDir: string;
  vadModelDir: string;
  runtimePack: string;
  stubArgsPath: string;
  service: OfflineTranscriptionService;
}

function fixture(workerSource = STUB_WORKER, opts: { offlineModel?: boolean; vadModel?: boolean; streamingModel?: boolean } = {}): Fixture {
  const root = mkdtempSync(join(tmpdir(), "yulu-offline-transcription-"));
  roots.push(root);
  const scriptDir = join(root, "scripts");
  const configDir = join(root, "config");
  const runtimePack = join(configDir, "local-caption/YuluLocalCaptionRuntime.bundle");
  const sitePackages = join(runtimePack, "Contents/Resources/site-packages");
  const streamingModelDir = join(configDir, "models/sherpa-onnx-streaming-paraformer-bilingual-zh-en");
  const offlineModelDir = join(configDir, "models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16");
  const vadModelDir = join(configDir, "models/silero-vad");
  mkdirSync(scriptDir, { recursive: true });
  mkdirSync(sitePackages, { recursive: true });
  mkdirSync(streamingModelDir, { recursive: true });
  mkdirSync(offlineModelDir, { recursive: true });
  mkdirSync(vadModelDir, { recursive: true });
  writeFileSync(join(scriptDir, "sherpa_caption_worker.py"), "");
  const stubArgsPath = join(root, "worker-args.json");
  const workerPath = join(scriptDir, "sherpa_offline_worker.py");
  writeFileSync(workerPath, workerSource);
  chmodSync(workerPath, 0o755);
  if (opts.offlineModel !== false) {
    for (const name of ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]) {
      writeFileSync(join(offlineModelDir, name), name);
    }
  }
  if (opts.streamingModel !== false) {
    for (const name of ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]) {
      writeFileSync(join(streamingModelDir, name), name);
    }
  }
  if (opts.vadModel !== false) {
    writeFileSync(join(vadModelDir, "silero_vad.onnx"), "vad");
  }
  process.env.YULU_PYTHON = pythonBin();
  process.env.STUB_ARGS_PATH = stubArgsPath;
  const config = {
    read: () => ({ transcription: { local: { final_model: "fire-red", offline_threads: 2 } } }),
  };
  const service = new OfflineTranscriptionService({ scriptDir, configDir, config: config as never });
  return { root, scriptDir, configDir, offlineModelDir, vadModelDir, runtimePack, stubArgsPath, service };
}

function makeStereoWav(root: string, frames = 1_600): string {
  const header = Buffer.alloc(44);
  const dataBytes = frames * 4;
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + dataBytes, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(2, 22);
  header.writeUInt32LE(16_000, 24);
  header.writeUInt32LE(64_000, 28);
  header.writeUInt16LE(4, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(dataBytes, 40);
  const path = join(root, "meeting.wav");
  writeFileSync(path, Buffer.concat([header, Buffer.alloc(dataBytes)]));
  return path;
}

describe("offlineThreadCount", () => {
  it("defaults to min(4, cores-2) with a floor of 2", () => {
    expect(offlineThreadCount()).toBe(Math.max(2, Math.min(4, cpus().length - 2)));
  });

  it("honours transcription.local.offline_threads when >= 1", () => {
    const config = { read: () => ({ transcription: { local: { offline_threads: 3 } } }) };
    expect(offlineThreadCount(config as never)).toBe(3);
    const disabled = { read: () => ({ transcription: { local: { offline_threads: 0 } } }) };
    expect(offlineThreadCount(disabled as never)).toBe(Math.max(2, Math.min(4, cpus().length - 2)));
  });
});

describe("OfflineTranscriptionService", () => {
  it("resamples both sources, spawns the worker with the expected protocol, and merges duplicate segments", async () => {
    const fx = fixture();
    const audioPath = makeStereoWav(fx.root);
    const onProgress = vi.fn();

    const result = await fx.service.transcribeFile(audioPath, "zh", onProgress);

    expect(result).toEqual({
      transcript: expect.any(String),
      provider: "sherpa-onnx-fire-red-asr-large-int8",
      chunks: 4,
      language: "zh",
    });
    // The duplicated cross-source text survives exactly once.
    expect(result.transcript.split("会议开始").length - 1).toBe(1);
    expect(result.transcript).toContain("讨论预算");
    expect(result.transcript).toContain("系统提示");
    const phases = onProgress.mock.calls.map((call) => call[0]!.phase);
    expect(phases).toContain("resample");
    expect(phases).toContain("vad");
    expect(phases).toContain("decode");

    const args = JSON.parse(readFileSync(fx.stubArgsPath, "utf8")) as string[];
    // sys.argv excludes the interpreter flags (-B -I -S) and the script path.
    expect(args).toEqual([
      "--runtime-pack",
      fx.runtimePack,
      "--model-dir",
      fx.offlineModelDir,
      "--vad-model",
      join(fx.vadModelDir, "silero_vad.onnx"),
      "--wav",
      `mic=${join(`${audioPath}.offline-tmp`, "mic.wav")}`,
      "--wav",
      `system=${join(`${audioPath}.offline-tmp`, "system.wav")}`,
      "--threads",
      "2",
      "--language",
      "zh",
    ]);

    // Staging is removed after the run.
    expect(existsSync(`${audioPath}.offline-tmp`)).toBe(false);
  });

  it("maps auto language and skips the system wav for mono input", async () => {
    const fx = fixture();
    const header = Buffer.alloc(44);
    header.write("RIFF", 0, "ascii");
    header.writeUInt32LE(36 + 3_200, 4);
    header.write("WAVE", 8, "ascii");
    header.write("fmt ", 12, "ascii");
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(1, 22);
    header.writeUInt32LE(16_000, 24);
    header.writeUInt32LE(32_000, 28);
    header.writeUInt16LE(2, 32);
    header.writeUInt16LE(16, 34);
    header.write("data", 36, "ascii");
    header.writeUInt32LE(3_200, 40);
    const monoPath = join(fx.root, "mono.wav");
    writeFileSync(monoPath, Buffer.concat([header, Buffer.alloc(3_200)]));

    await fx.service.transcribeFile(monoPath, "auto");

    const args = JSON.parse(readFileSync(fx.stubArgsPath, "utf8")) as string[];
    const wavs = args.filter((_, index) => args[index - 1] === "--wav");
    expect(wavs).toHaveLength(1);
    expect(wavs[0]).toContain("mic=");
    expect(args[args.indexOf("--language") + 1]).toBe("auto");
  });

  it("rejects Japanese with a visible error", async () => {
    const fx = fixture();
    const audioPath = makeStereoWav(fx.root);
    await expect(fx.service.transcribeFile(audioPath, "ja"))
      .rejects.toThrow(AgentUnavailableError);
    await expect(fx.service.transcribeFile(audioPath, "ja")).rejects.toThrow("仅支持中英文");
  });

  it("keeps final-tier transcription working when the streaming model is uninstalled", async () => {
    const fx = fixture(undefined, { streamingModel: false });
    const audioPath = makeStereoWav(fx.root);

    const result = await fx.service.transcribeFile(audioPath, "zh");

    expect(result.provider).toBe("sherpa-onnx-fire-red-asr-large-int8");
    expect(result.transcript).toContain("讨论预算");
    expect(fx.service.modelInstalled()).toBe(true);
  });

  it("rejects with an actionable message when the offline assets are missing", async () => {
    const fx = fixture(STUB_WORKER, { offlineModel: false });
    const audioPath = makeStereoWav(fx.root);
    await expect(fx.service.transcribeFile(audioPath, "zh"))
      .rejects.toThrow("离线高质量模型未安装");
  });

  it("reports missing VAD assets the same way", async () => {
    const fx = fixture(STUB_WORKER, { vadModel: false });
    const audioPath = makeStereoWav(fx.root);
    await expect(fx.service.transcribeFile(audioPath, "zh"))
      .rejects.toThrow("离线高质量模型未安装");
  });

  it("surfaces worker stderr fatal messages", async () => {
    const fatalWorker = [
      "#!/usr/bin/env python3",
      "import json, sys",
      "print(json.dumps({'fatal': '离线模型不完整: tokens.txt'}), file=sys.stderr, flush=True)",
      "sys.exit(2)",
      "",
    ].join("\n");
    const fx = fixture(fatalWorker);
    const audioPath = makeStereoWav(fx.root);
    await expect(fx.service.transcribeFile(audioPath, "zh")).rejects.toThrow("离线模型不完整");
    expect(existsSync(`${audioPath}.offline-tmp`)).toBe(false);
  });

  it("kills a stuck worker at the timeout and fails visibly", async () => {
    const fx = fixture(HANG_WORKER);
    const audioPath = makeStereoWav(fx.root);
    const realSetTimeout = globalThis.setTimeout;
    vi.spyOn(globalThis, "setTimeout").mockImplementation((((handler: (...args: never[]) => void, ms?: number, ...rest: never[]) =>
      realSetTimeout(handler, Math.min(ms ?? 0, 200), ...rest)) as unknown) as typeof setTimeout);

    await expect(fx.service.transcribeFile(audioPath, "zh")).rejects.toThrow("离线转录超时");
    expect(existsSync(`${audioPath}.offline-tmp`)).toBe(false);
  });
});
