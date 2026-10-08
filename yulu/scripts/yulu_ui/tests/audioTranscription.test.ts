import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { AudioTranscriptionService } from "../src/audioTranscription.js";
import { AgentUnavailableError } from "../src/agentGateway.js";
import { buildGlossaryContract } from "../src/glossaryContract.js";

function setup(
  engine: "local" | "xai",
  xaiConsent = true,
  localConfig: Record<string, unknown> = {},
  offline?: { transcribeFile: ReturnType<typeof vi.fn> },
) {
  const configValue = {
    transcription: {
      engine,
      local: { final_model: "paraformer-replay", ...localConfig },
    },
  };
  const config = { read: () => configValue };
  const local = {
    provider: "local-test",
    status: vi.fn(() => ({ ready: true, error: "" })),
    warm: vi.fn(async () => {}),
    start: vi.fn(async () => {}),
    feed: vi.fn(async () => ({ updates: {} })),
    finish: vi.fn(async () => ({ updates: {} })),
    abort: vi.fn(async () => {}),
    close: vi.fn(async () => {}),
  };
  const xai = {
    provider: "xai-oauth:yulu",
    credentialStatus: vi.fn(() => ({ connected: true, detail: "connected" })),
    warm: vi.fn(async () => {}),
    start: vi.fn(async () => {}),
    feed: vi.fn(async () => ({ updates: {} })),
    finish: vi.fn(async () => ({ updates: {} })),
    abort: vi.fn(async () => {}),
    close: vi.fn(async () => {}),
    transcribeFile: vi.fn(async () => ({ transcript: "xAI transcript", provider: "xai-oauth:yulu", chunks: 1, language: "zh" as const })),
    testCredential: vi.fn(async () => ({ ok: true as const, provider: "xai-oauth:yulu" })),
  };
  const service = new AudioTranscriptionService(
    config as never,
    local as never,
    xai as never,
    () => xaiConsent,
    () => buildGlossaryContract([{ term: "Agent Key", canonical: "AgentKey", scope: "both" }]),
    offline as never,
  );
  return { configValue, local, xai, service };
}

function makeMonoWav(samples = 1_600): string {
  const dir = mkdtempSync(join(tmpdir(), "yulu-audio-replay-"));
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + samples * 2, 4);
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
  header.writeUInt32LE(samples * 2, 40);
  writeFileSync(join(dir, "Demo_20260805_140009.wav"), Buffer.concat([header, Buffer.alloc(samples * 2)]));
  return join(dir, "Demo_20260805_140009.wav");
}

describe("AudioTranscriptionService", () => {
  it("includes the live glossary in selected xAI realtime sessions", async () => {
    const { xai, service } = setup("xai");
    await service.start("zh");
    expect(xai.start).toHaveBeenCalledWith("zh", expect.objectContaining({ prompt: "AgentKey" }));
    await service.abort();
  });

  it("rejects realtime and final xAI audio processing without current disclosure consent", async () => {
    const { local, xai, service } = setup("xai", false);

    await expect(service.start("zh")).rejects.toThrow("current Cloud Transcription Consent");
    await expect(service.transcribeFile("/tmp/not-opened.wav", "zh"))
      .rejects.toThrow("current Cloud Transcription Consent");
    expect(xai.start).not.toHaveBeenCalled();
    expect(xai.transcribeFile).not.toHaveBeenCalled();
    expect(local.start).not.toHaveBeenCalled();
  });

  it("does not require cloud consent for selected local transcription", async () => {
    const { local, xai, service } = setup("local", false);

    await service.start("zh");
    expect(local.start).toHaveBeenCalledOnce();
    expect(xai.start).not.toHaveBeenCalled();
  });

  it("does not fall back to local when the selected xAI engine fails", async () => {
    const { local, xai, service } = setup("xai");
    xai.transcribeFile.mockRejectedValueOnce(new Error("xAI offline"));

    await expect(service.transcribeFile("/tmp/not-opened.wav", "zh")).rejects.toThrow("xAI offline");
    expect(local.start).not.toHaveBeenCalled();
  });

  it("uses a finished trusted realtime transcript when final xAI transcription fails", async () => {
    const dir = mkdtempSync(join(tmpdir(), "yulu-audio-transcription-"));
    const audioPath = join(dir, "Demo_20260805_140009.wav");
    writeFileSync(audioPath.replace(/\.wav$/, ".realtime.transcript.txt"), "完整的实时转写");
    writeFileSync(audioPath.replace(/\.wav$/, ".realtime.coverage.json"), JSON.stringify({
      provider: "xai-oauth:yulu",
      chunks: 231,
      trusted: true,
      finished: true,
    }));
    const { local, xai, service } = setup("xai");
    xai.transcribeFile.mockRejectedValueOnce(new Error("xAI returned an empty transcript"));

    try {
      await expect(service.transcribeFile(audioPath, "zh")).resolves.toEqual({
        transcript: "完整的实时转写",
        provider: "xai-oauth:yulu",
        chunks: 231,
        language: "zh",
      });
      expect(local.start).not.toHaveBeenCalled();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("does not fall back to xAI when the selected local engine is unavailable", async () => {
    const { local, xai, service } = setup("local");
    local.status.mockReturnValueOnce({ ready: false, error: "local model missing" });

    await expect(service.transcribeFile("/tmp/not-opened.wav", "zh")).rejects.toThrow("local model missing");
    expect(xai.transcribeFile).not.toHaveBeenCalled();
  });

  it("locks a realtime session to the engine selected at start", async () => {
    const { configValue, local, xai, service } = setup("local");
    await service.start("zh");
    configValue.transcription.engine = "xai";
    await service.feed({ mic: Buffer.alloc(320) });
    await service.finish();

    expect(local.start).toHaveBeenCalledOnce();
    expect(local.feed).toHaveBeenCalledOnce();
    expect(local.finish).toHaveBeenCalledOnce();
    expect(xai.start).not.toHaveBeenCalled();
  });

  it("delegates local final transcription to the offline FireRedASR service", async () => {
    const offlineResult = {
      transcript: "离线高质量结果",
      provider: "sherpa-onnx-fire-red-asr-large-int8",
      chunks: 3,
      language: "zh" as const,
    };
    const offline = { transcribeFile: vi.fn(async () => offlineResult) };
    const { local, service } = setup("local", true, { final_model: "fire-red" }, offline);
    const audioPath = makeMonoWav();
    const onProgress = vi.fn();

    await expect(service.transcribeFile(audioPath, "zh", undefined, "final", onProgress))
      .resolves.toEqual(offlineResult);
    expect(offline.transcribeFile).toHaveBeenCalledWith(audioPath, "zh", onProgress);
    expect(local.start).not.toHaveBeenCalled();
  });

  it("fails visibly when fire-red is the final model but its assets are missing", async () => {
    const offline = {
      transcribeFile: vi.fn(async () => {
        throw new AgentUnavailableError("离线高质量模型未安装：请到设置安装离线高质量转录模型");
      }),
    };
    const { local, service } = setup("local", true, { final_model: "fire-red" }, offline);

    await expect(service.transcribeFile("/tmp/not-opened.wav", "zh"))
      .rejects.toThrow("离线高质量模型未安装");
    expect(local.start).not.toHaveBeenCalled();
  });

  it("keeps the streaming replay path for paraformer-replay final model", async () => {
    const { local, service } = setup("local", true, { final_model: "paraformer-replay" });
    local.feed.mockResolvedValue({
      updates: { mic: { partial: "", stable: [{ text: "本地重放结果", endMs: 800 }] } },
    });
    const audioPath = makeMonoWav();

    await expect(service.transcribeFile(audioPath, "zh")).resolves.toMatchObject({
      transcript: "本地重放结果",
      provider: "local-test",
    });
    expect(local.start).toHaveBeenCalledWith("zh");
    expect(local.finish).toHaveBeenCalledOnce();
  });

  it("uses the streaming replay path for the fast tier even with fire-red configured", async () => {
    const offline = { transcribeFile: vi.fn(async () => ({ transcript: "离线", provider: "offline", chunks: 1 })) };
    const { local, service } = setup("local", true, { final_model: "fire-red" }, offline);
    local.feed.mockResolvedValue({
      updates: { mic: { partial: "", stable: [{ text: "快速重放结果", endMs: 400 }] } },
    });
    const audioPath = makeMonoWav();

    await expect(service.transcribeFile(audioPath, "zh", undefined, "fast")).resolves.toMatchObject({
      transcript: "快速重放结果",
      provider: "local-test",
    });
    expect(offline.transcribeFile).not.toHaveBeenCalled();
    expect(local.start).toHaveBeenCalledOnce();
  });
});
