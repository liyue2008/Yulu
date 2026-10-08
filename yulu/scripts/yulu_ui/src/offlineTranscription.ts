import { spawn } from "node:child_process";
import { closeSync, mkdirSync, openSync, readSync, rmSync, statSync, writeSync } from "node:fs";
import { cpus } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import type { ConfigManager } from "./config.js";
import { AgentUnavailableError } from "./agentGateway.js";
import {
  isOfflineModelInstalled,
  isVadModelInstalled,
  OFFLINE_MODEL_PROVIDER,
  resolveLocalCaptionRuntime,
  type LocalCaptionRuntime,
} from "./localCaptionEngine.js";
import { SourceSeparatedResampler } from "./pcmResampler.js";
import {
  captionsLikelyDuplicate,
  cleanTranscriptText,
  dedupeTranscriptSegment,
  parseWavFormat,
  type TranscriptionLanguage,
  type TranscriptionResult,
} from "./realtimeTranscription.js";

const WORKER_TIMEOUT_MS = 90 * 60_000;
const RESAMPLE_CHUNK_FRAMES = 32_768;

export interface OfflineTranscriptionProgress {
  phase: "resample" | "vad" | "decode";
  source?: string;
  percent: number | null;
  message: string;
}

export type OfflineProgressListener = (progress: OfflineTranscriptionProgress) => void;

/** Default worker threads: min(4, cores-2), floor 2; transcription.local.offline_threads overrides. */
export function offlineThreadCount(config?: ConfigManager): number {
  const configured = config?.read().transcription.local.offline_threads;
  if (configured && configured >= 1) return configured;
  return Math.max(2, Math.min(4, cpus().length - 2));
}

function workerLanguage(language: TranscriptionLanguage): "zh" | "en" | "auto" {
  if (language === "en") return "en";
  if (language === "auto") return "auto";
  return "zh";
}

interface OfflineSegment {
  source: "mic" | "system";
  startMs: number;
  endMs: number;
  text: string;
}

interface WorkerResultEvent {
  provider?: string;
  sources?: Record<string, Array<{ startMs?: unknown; endMs?: unknown; text?: unknown }>>;
}

/** One-shot local final transcription via the FireRedASR offline worker. */
export class OfflineTranscriptionService {
  constructor(private readonly options: {
    scriptDir: string;
    configDir: string;
    modelsDir?: string;
    legacyConfigDir?: string;
    legacyModelsDir?: string;
    config: ConfigManager;
  }) {}

  modelInstalled(): boolean {
    const runtime = this.resolveRuntime();
    if (!runtime) return false;
    return isOfflineModelInstalled(runtime.offlineModelDir) && isVadModelInstalled(runtime.vadModelDir);
  }

  async transcribeFile(
    audioPath: string,
    language: TranscriptionLanguage,
    onProgress?: OfflineProgressListener,
  ): Promise<TranscriptionResult> {
    if (language === "ja") {
      throw new AgentUnavailableError("本地离线模型仅支持中英文；日语请选择 xAI 引擎");
    }
    const runtime = this.resolveRuntime();
    if (!runtime) {
      throw new AgentUnavailableError("本地识别运行时未安装：请到设置安装本地音频引擎，或改用 xAI 引擎");
    }
    if (!isOfflineModelInstalled(runtime.offlineModelDir) || !isVadModelInstalled(runtime.vadModelDir)) {
      throw new AgentUnavailableError(
        "离线高质量模型未安装：请到设置安装离线高质量转录模型，或把 transcription.local.final_model 改为 paraformer-replay",
      );
    }
    const staging = `${audioPath}.offline-tmp`;
    mkdirSync(staging, { recursive: true });
    try {
      const wavPaths = await this.resampleToStaging(audioPath, staging, onProgress);
      const segments = await this.runWorker(runtime, wavPaths, language, onProgress);
      const transcript = mergeOfflineSegments(segments);
      if (!transcript) throw new Error("本地离线转写没有识别到语音");
      return { transcript, provider: OFFLINE_MODEL_PROVIDER, chunks: Math.max(1, segments.length), language };
    } finally {
      rmSync(staging, { recursive: true, force: true });
    }
  }

  private resolveRuntime(): LocalCaptionRuntime | null {
    return resolveLocalCaptionRuntime({
      scriptDir: this.options.scriptDir,
      configDir: this.options.configDir,
      modelsDir: this.options.modelsDir,
      legacyConfigDir: this.options.legacyConfigDir,
      legacyModelsDir: this.options.legacyModelsDir,
    });
  }

  /** Resample each source channel to its own 16 kHz mono WAV inside the staging dir. */
  private async resampleToStaging(
    audioPath: string,
    staging: string,
    onProgress?: OfflineProgressListener,
  ): Promise<{ mic: string; system?: string }> {
    const format = parseWavFormat(audioPath);
    if (format.channels < 1 || format.channels > 2) {
      throw new Error(`不支持的声道数: ${format.channels}`);
    }
    const resampler = new SourceSeparatedResampler(format);
    const writers = [
      new WavStreamWriter(join(staging, "mic.wav")),
      ...(format.channels >= 2 ? [new WavStreamWriter(join(staging, "system.wav"))] : []),
    ];
    const size = statSync(audioPath).size;
    const fd = openSync(audioPath, "r");
    const buffer = Buffer.alloc(RESAMPLE_CHUNK_FRAMES * format.blockAlign);
    let offset = format.dataOffset;
    try {
      while (offset < size) {
        const read = readSync(fd, buffer, 0, buffer.length, offset);
        if (read <= 0) break;
        offset += read;
        const aligned = buffer.subarray(0, read - (read % format.blockAlign));
        const separated = resampler.feed(aligned);
        if (separated.chunks.mic?.length) writers[0]!.write(separated.chunks.mic);
        if (writers[1] && separated.chunks.system?.length) writers[1].write(separated.chunks.system);
        const percent = Math.min(99, Math.floor(((offset - format.dataOffset) / Math.max(1, size - format.dataOffset)) * 100));
        onProgress?.({ phase: "resample", percent, message: `正在准备离线转写音频（${percent}%）` });
        await new Promise<void>((resolve) => setImmediate(resolve));
      }
    } finally {
      closeSync(fd);
      for (const writer of writers) writer.close();
    }
    return {
      mic: writers[0]!.path,
      ...(writers[1] ? { system: writers[1].path } : {}),
    };
  }

  private runWorker(
    runtime: LocalCaptionRuntime,
    wavPaths: { mic: string; system?: string },
    language: TranscriptionLanguage,
    onProgress?: OfflineProgressListener,
  ): Promise<OfflineSegment[]> {
    const script = join(this.options.scriptDir, "sherpa_offline_worker.py");
    return new Promise<OfflineSegment[]>((resolve, reject) => {
      const child = spawn(runtime.python, [
        "-B",
        "-I",
        "-S",
        script,
        "--runtime-pack", runtime.runtimePack,
        "--model-dir", runtime.offlineModelDir,
        "--vad-model", join(runtime.vadModelDir, "silero_vad.onnx"),
        "--wav", `mic=${wavPaths.mic}`,
        ...(wavPaths.system ? ["--wav", `system=${wavPaths.system}`] : []),
        "--threads", String(offlineThreadCount(this.options.config)),
        "--language", workerLanguage(language),
      ], {
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONDONTWRITEBYTECODE: "1",
          PYTHONNOUSERSITE: "1",
          PYTHONUNBUFFERED: "1",
          OMP_NUM_THREADS: "1",
        },
      });
      let settled = false;
      let stderr = "";
      let result: WorkerResultEvent | null = null;
      const finish = (error: Error | null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (error) reject(error);
        else resolve(collectSegments(result));
      };
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        finish(new Error("离线转录超时（90 分钟）已终止"));
      }, WORKER_TIMEOUT_MS);
      const lines = createInterface({ input: child.stdout });
      lines.on("line", (line) => {
        try {
          const event = JSON.parse(line) as Record<string, unknown>;
          if (event.event === "progress") {
            onProgress?.({
              phase: event.phase === "decode" ? "decode" : "vad",
              source: typeof event.source === "string" ? event.source : undefined,
              percent: typeof event.percent === "number" ? event.percent : null,
              message: typeof event.message === "string" ? event.message : "",
            });
          }
          if (event.event === "result") result = event as WorkerResultEvent;
        } catch { /* ignore non-protocol output */ }
      });
      child.stderr.on("data", (chunk: Buffer) => { stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4_000); });
      child.once("error", (error) => finish(error as Error));
      child.once("exit", (code) => {
        let fatal = "";
        try {
          const parsed = JSON.parse(stderr.trim().split("\n").pop() ?? "") as { fatal?: string };
          fatal = parsed.fatal ?? "";
        } catch { /* plain-text stderr */ }
        if (code === 0 && result) finish(null);
        else finish(new Error(fatal || stderr.trim() || `离线转录进程退出（${code}）`));
      });
    });
  }
}

function collectSegments(result: WorkerResultEvent | null): OfflineSegment[] {
  const segments: OfflineSegment[] = [];
  for (const source of ["mic", "system"] as const) {
    const list = result?.sources?.[source];
    if (!Array.isArray(list)) continue;
    for (const item of list) {
      const text = cleanTranscriptText(typeof item.text === "string" ? item.text : "");
      if (!text) continue;
      segments.push({
        source,
        startMs: Number(item.startMs ?? 0) || 0,
        endMs: Number(item.endMs ?? 0) || 0,
        text,
      });
    }
  }
  return segments;
}

/** Cross-source merge: overlapping duplicates keep the system version (cleaner meeting audio). */
function mergeOfflineSegments(segments: OfflineSegment[]): string {
  const ordered = [...segments].sort((a, b) => a.startMs - b.startMs || a.endMs - b.endMs);
  const kept: OfflineSegment[] = [];
  for (const candidate of ordered) {
    const duplicate = kept.findIndex((existing) =>
      existing.source !== candidate.source &&
      candidate.startMs < existing.endMs + 1_500 &&
      candidate.endMs > existing.startMs - 1_500 &&
      captionsLikelyDuplicate(existing.text, candidate.text));
    if (duplicate >= 0) {
      if (candidate.source === "system" && kept[duplicate]!.source === "mic") kept.splice(duplicate, 1, candidate);
      continue;
    }
    kept.push(candidate);
  }
  let transcript = "";
  for (const item of kept) {
    const addition = dedupeTranscriptSegment(transcript, item.text);
    if (addition) transcript = cleanTranscriptText([transcript, addition].filter(Boolean).join("\n"));
  }
  return transcript;
}

/** Streaming 16 kHz mono PCM16 WAV writer; patches RIFF/data sizes on close. */
class WavStreamWriter {
  private readonly fd: number;
  private bytes = 0;

  constructor(readonly path: string) {
    this.fd = openSync(path, "w");
    const header = Buffer.alloc(44);
    header.write("RIFF", 0, "ascii");
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
    writeSync(this.fd, header);
  }

  write(pcm: Buffer): void {
    if (pcm.length === 0) return;
    writeSync(this.fd, pcm);
    this.bytes += pcm.length;
  }

  close(): void {
    try {
      const riffSize = Buffer.alloc(4);
      riffSize.writeUInt32LE(36 + this.bytes, 0);
      writeSync(this.fd, riffSize, 0, riffSize.length, 4);
      const dataSize = Buffer.alloc(4);
      dataSize.writeUInt32LE(this.bytes, 0);
      writeSync(this.fd, dataSize, 0, dataSize.length, 40);
    } finally {
      closeSync(this.fd);
    }
  }
}

/** Resample each source channel to its own 16 kHz mono WAV inside the staging dir. */