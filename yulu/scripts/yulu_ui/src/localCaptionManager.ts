import { spawn } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import {
  isOfflineModelInstalled,
  isVadModelInstalled,
  OFFLINE_MODEL_PROVIDER,
  resolveLocalCaptionAssets,
  resolveLocalCaptionRuntime,
  SherpaCaptionEngine,
  type CaptionSource,
  type StreamingCaptionEngine,
  type StreamingCaptionUpdate,
} from "./localCaptionEngine.js";

export type LocalCaptionModelKey = "streaming" | "offline-final" | "silero-vad";

export interface LocalCaptionStatus {
  installed: boolean;
  ready: boolean;
  provider: string;
  model: string;
  runtimeBytes: number;
  modelBytes: number;
  offlineModelReady: boolean;
  offlineModelBytes: number;
  vadReady: boolean;
  operation: "idle" | "installing" | "uninstalling" | "testing";
  phase: string | null;
  percent: number | null;
  message: string | null;
  error: string | null;
  sessionActive: boolean;
}

const MODEL_NAME = "sherpa-onnx-streaming-paraformer-bilingual-zh-en";

function directoryBytes(path: string): number {
  if (!existsSync(path)) return 0;
  let total = 0;
  const stack = [path];
  while (stack.length > 0) {
    const current = stack.pop()!;
    try {
      for (const name of readdirSync(current)) {
        const item = join(current, name);
        try {
          const stat = statSync(item);
          if (stat.isDirectory()) stack.push(item);
          else if (stat.isFile()) total += stat.size;
        } catch { /* an uninstall may remove a file while status is being read */ }
      }
    } catch { /* an uninstall may remove a directory while status is being read */ }
  }
  return total;
}

function bootstrapPython(): string {
  const configured = process.env.YULU_PYTHON?.trim();
  if (configured && existsSync(configured)) return configured;
  throw new Error("Application Runtime 内置 Python 不可用");
}

export class LocalCaptionManager implements StreamingCaptionEngine {
  private engine: SherpaCaptionEngine | null = null;
  private active = false;
  private operation: LocalCaptionStatus["operation"] = "idle";
  private phase: string | null = null;
  private percent: number | null = null;
  private message: string | null = null;
  private error: string | null = null;

  constructor(private readonly options: {
    scriptDir: string;
    configDir: string;
    modelsDir?: string;
    legacyConfigDir?: string;
    legacyModelsDir?: string;
    selected: () => boolean;
  }) {}

  get provider(): string {
    return "sherpa-onnx-paraformer-int8";
  }

  status(): LocalCaptionStatus {
    const runtime = this.resolveRuntime();
    const modelsDir = this.options.modelsDir ?? join(this.options.configDir, "models");
    const assets = runtime ?? resolveLocalCaptionAssets({
      configDir: this.options.configDir,
      modelsDir: this.options.modelsDir,
    });
    return {
      installed: runtime !== null,
      ready: runtime !== null && this.error === null,
      provider: this.provider,
      model: MODEL_NAME,
      runtimeBytes: directoryBytes(runtime ? dirname(runtime.runtimePack) : join(this.options.configDir, "local-caption")),
      modelBytes: directoryBytes(runtime?.modelDir ?? join(modelsDir, MODEL_NAME)),
      offlineModelReady: isOfflineModelInstalled(assets.offlineModelDir),
      offlineModelBytes: directoryBytes(assets.offlineModelDir),
      vadReady: isVadModelInstalled(assets.vadModelDir),
      operation: this.operation,
      phase: this.phase,
      percent: this.percent,
      message: this.message,
      error: this.error,
      sessionActive: this.active,
    };
  }

  async install(input?: { model?: LocalCaptionModelKey }): Promise<LocalCaptionStatus> {
    if (this.operation !== "idle") throw new Error("本地模型已有操作正在进行");
    const model = input?.model;
    this.setOperation("installing", "runtime", model === "offline-final" ? "正在下载离线高质量转录模型（约 1.5 GB）" : "正在准备本地识别运行时");
    try {
      await this.runInstaller("install", model);
      this.error = null;
      if (model === "offline-final") this.message = "离线高质量转录模型已安装";
      else if (model === "silero-vad") this.message = "语音活动检测模型已安装";
      else this.message = "本地实时转录模型已安装";
      if (!model && this.options.selected()) await this.warm();
    } catch (error) {
      this.error = (error as Error).message;
      throw error;
    } finally {
      this.operation = "idle";
      this.phase = null;
      this.percent = null;
    }
    return this.status();
  }

  async uninstall(input?: { model?: LocalCaptionModelKey }): Promise<LocalCaptionStatus> {
    if (this.active) throw new Error("录音进行中，不能卸载本地模型");
    if (this.operation !== "idle") throw new Error("本地模型已有操作正在进行");
    const model = input?.model;
    this.setOperation("uninstalling", "cleanup", "正在移除本地模型");
    try {
      if (!model) {
        await this.engine?.close();
        this.engine = null;
      }
      await this.runInstaller("uninstall", model);
      this.error = null;
      if (model === "offline-final") this.message = "离线高质量转录模型已移除";
      else if (model === "silero-vad") this.message = "语音活动检测模型已移除";
      else this.message = "本地模型已移除；重新安装前，本地音频引擎不可用";
    } catch (error) {
      this.error = (error as Error).message;
      throw error;
    } finally {
      this.operation = "idle";
      this.phase = null;
      this.percent = null;
    }
    return this.status();
  }

  async test(): Promise<{ ok: true; provider: string; loadMs: number }> {
    if (this.operation !== "idle") throw new Error("本地模型已有操作正在进行");
    this.setOperation("testing", "warmup", "正在加载并测试本地模型");
    const started = performance.now();
    try {
      await this.ensureEngine().warm();
      const loadMs = Math.round((performance.now() - started) * 100) / 100;
      this.error = null;
      this.message = `本地模型测试通过（${loadMs} ms）`;
      return { ok: true, provider: this.provider, loadMs };
    } catch (error) {
      this.error = (error as Error).message;
      throw error;
    } finally {
      this.operation = "idle";
      this.phase = null;
      this.percent = null;
    }
  }

  /** Offline-model smoke test: run the one-shot FireRedASR worker on a tiny synthetic WAV. */
  async testOffline(): Promise<{ ok: true; provider: string; loadMs: number; segments: number }> {
    if (this.operation !== "idle") throw new Error("本地模型已有操作正在进行");
    const runtime = this.resolveRuntime();
    const assets = runtime ?? resolveLocalCaptionAssets({
      configDir: this.options.configDir,
      modelsDir: this.options.modelsDir,
    });
    if (!isOfflineModelInstalled(assets.offlineModelDir) || !isVadModelInstalled(assets.vadModelDir)) {
      throw new Error("离线高质量转录模型尚未安装；请先在设置中安装");
    }
    if (!runtime) throw new Error("本地识别运行时未安装；请先安装本地音频引擎");
    this.setOperation("testing", "warmup", "正在加载离线高质量转录模型（约 1.6 GB）");
    const started = performance.now();
    try {
      const result = await this.runOfflineWorker(runtime, assets);
      const loadMs = Math.round((performance.now() - started) * 100) / 100;
      this.error = null;
      this.message = `离线高质量转录模型测试通过（${loadMs} ms）`;
      return { ok: true, provider: OFFLINE_MODEL_PROVIDER, loadMs, segments: result };
    } catch (error) {
      this.error = (error as Error).message;
      throw error;
    } finally {
      this.operation = "idle";
      this.phase = null;
      this.percent = null;
    }
  }

  async syncSelection(): Promise<void> {
    if (!this.options.selected()) {
      if (!this.active) {
        await this.engine?.close();
        this.engine = null;
      }
      return;
    }
    const runtime = this.resolveRuntime();
    if (runtime) await this.warm();
  }

  async warm(): Promise<void> {
    if (!this.options.selected()) throw new Error("当前未选择本地音频引擎");
    try {
      await this.ensureEngine().warm();
      this.error = null;
    } catch (error) {
      this.error = (error as Error).message;
      throw error;
    }
  }

  async start(language: "zh" | "en" | "ja" | "auto"): Promise<void> {
    if (!this.options.selected()) throw new Error("当前未选择本地音频引擎");
    if (language === "ja") throw new Error("本地 Paraformer 仅支持中英文；如需日语，请在设置中明确选择 xAI 云端");
    try {
      await this.ensureEngine().start(language);
      this.error = null;
      this.active = true;
    } catch (error) {
      this.error = (error as Error).message;
      throw error;
    }
  }

  async feed(chunks: Partial<Record<CaptionSource, Buffer>>): Promise<StreamingCaptionUpdate> {
    if (!this.active) throw new Error("本地实时转录会话未启动");
    try { return await this.ensureEngine().feed(chunks); }
    catch (error) {
      this.error = (error as Error).message;
      throw error;
    }
  }

  async finish(): Promise<StreamingCaptionUpdate> {
    if (!this.active) return { updates: {} };
    try { return await this.ensureEngine().finish(); }
    catch (error) {
      this.error = (error as Error).message;
      throw error;
    }
    finally {
      this.active = false;
      if (!this.options.selected()) {
        try { await this.engine?.close(); } catch { /* the recording already ended */ }
        this.engine = null;
      }
    }
  }

  async abort(): Promise<void> {
    try { await this.engine?.abort(); }
    finally {
      this.active = false;
      if (!this.options.selected()) {
        try { await this.engine?.close(); } catch { /* preserve the abort result */ }
        this.engine = null;
      }
    }
  }

  async close(): Promise<void> {
    this.active = false;
    await this.engine?.close();
    this.engine = null;
  }

  private ensureEngine(): SherpaCaptionEngine {
    if (this.engine) return this.engine;
    const runtime = this.resolveRuntime();
    if (!runtime) throw new Error("本地 sherpa-onnx 模型尚未安装");
    this.engine = new SherpaCaptionEngine(runtime);
    return this.engine;
  }

  private resolveRuntime() {
    return resolveLocalCaptionRuntime({
      scriptDir: this.options.scriptDir,
      configDir: this.options.configDir,
      modelsDir: this.options.modelsDir,
      legacyConfigDir: this.options.legacyConfigDir,
      legacyModelsDir: this.options.legacyModelsDir,
    });
  }

  private setOperation(operation: LocalCaptionStatus["operation"], phase: string, message: string): void {
    this.operation = operation;
    this.phase = phase;
    this.percent = null;
    this.message = message;
    this.error = null;
  }

  private runInstaller(action: "install" | "uninstall", model?: LocalCaptionModelKey): Promise<void> {
    const script = join(this.options.scriptDir, "local_caption_runtime.py");
    return new Promise((resolve, reject) => {
      const environment = { ...process.env };
      for (const key of Object.keys(environment)) {
        if (key.startsWith("PYTHON")) delete environment[key];
      }
      const child = spawn(bootstrapPython(), [
        "-I",
        "-S",
        "-B",
        script,
        action,
        "--config-dir",
        this.options.configDir,
        "--models-dir",
        this.options.modelsDir ?? join(this.options.configDir, "models"),
        ...(model ? ["--model", model] : []),
      ], {
        stdio: ["ignore", "pipe", "pipe"],
        env: environment,
      });
      let stderr = "";
      let finalError = "";
      const lines = createInterface({ input: child.stdout });
      lines.on("line", (line) => {
        try {
          const event = JSON.parse(line) as Record<string, unknown>;
          if (event.event === "progress") {
            this.phase = typeof event.phase === "string" ? event.phase : this.phase;
            this.message = typeof event.message === "string" ? event.message : this.message;
            this.percent = typeof event.percent === "number" ? event.percent : null;
          }
          if (event.event === "result" && event.ok === false) finalError = String(event.error ?? "安装失败");
        } catch { /* ignore non-protocol output */ }
      });
      child.stderr.on("data", (chunk: Buffer) => { stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4_000); });
      child.once("error", reject);
      child.once("exit", (code) => {
        lines.close();
        if (code === 0) resolve();
        else reject(new Error(finalError || stderr.trim() || `本地模型管理进程退出（${code}）`));
      });
    });
  }

  /** Spawn the one-shot offline worker on ~2s of synthetic silence and collect the result event. */
  private runOfflineWorker(
    runtime: NonNullable<ReturnType<LocalCaptionManager["resolveRuntime"]>>,
    assets: { offlineModelDir: string; vadModelDir: string },
  ): Promise<number> {
    const script = join(this.options.scriptDir, "sherpa_offline_worker.py");
    return new Promise<number>((resolve, reject) => {
      void (async () => {
        const staging = await mkdtemp(join(tmpdir(), "yulu-offline-smoke-"));
        const wav = join(staging, "smoke.wav");
        // 2s of 16 kHz mono silence: loads the offline model + VAD and runs the pipeline
        // end-to-end without depending on speech synthesis; expected 0 segments.
        const samples = Buffer.alloc(32_000, 0, "binary");
        const header = Buffer.alloc(44);
        header.write("RIFF", 0, "ascii");
        header.writeUInt32LE(36 + samples.length, 4);
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
        header.writeUInt32LE(samples.length, 40);
        await writeFile(wav, Buffer.concat([header, samples]));
        const environment = {
          ...process.env,
          PYTHONDONTWRITEBYTECODE: "1",
          PYTHONNOUSERSITE: "1",
          PYTHONUNBUFFERED: "1",
          OMP_NUM_THREADS: "1",
        };
        const child = spawn(runtime.python, [
          "-B",
          "-I",
          "-S",
          script,
          "--runtime-pack", runtime.runtimePack,
          "--model-dir", assets.offlineModelDir,
          "--vad-model", join(assets.vadModelDir, "silero_vad.onnx"),
          "--wav", `mic=${wav}`,
          "--threads", "4",
          "--language", "zh",
        ], { stdio: ["ignore", "pipe", "pipe"], env: environment });
        const timer = setTimeout(() => {
          child.kill("SIGKILL");
          reject(new Error("离线模型测试超时（10 分钟）"));
        }, 10 * 60_000);
        let settled = false;
        let stderr = "";
        let segments = -1;
        const finish = (error: Error | null) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          void rm(staging, { recursive: true, force: true }).catch(() => undefined);
          if (error) reject(error);
          else resolve(segments);
        };
        const lines = createInterface({ input: child.stdout });
        lines.on("line", (line) => {
          try {
            const event = JSON.parse(line) as Record<string, unknown>;
            if (event.event === "progress" && typeof event.message === "string") {
              this.message = event.message;
              if (typeof event.percent === "number") this.percent = event.percent;
            }
            if (event.event === "result") {
              const sources = event.sources as Record<string, Array<unknown>> | undefined;
              segments = Object.values(sources ?? {}).reduce((total, list) => total + (Array.isArray(list) ? list.length : 0), 0);
            }
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
          if (code === 0 && segments >= 0) finish(null);
          else finish(new Error(fatal || stderr.trim() || `离线转录测试进程退出（${code}）`));
        });
      })().catch(reject);
    });
  }
}
