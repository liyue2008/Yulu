import { describe, expect, it, vi } from "vitest";
import { createCaller } from "../../src/trpc.js";
import { localCaptionRouter } from "../../src/routers/localCaption.js";

describe("localCaption router", () => {
  it("exposes model state and delegates lifecycle actions", async () => {
    const state = { installed: true, ready: true, operation: "idle" };
    const localCaption = {
      status: vi.fn(() => state),
      install: vi.fn(async () => state),
      uninstall: vi.fn(async () => ({ ...state, installed: false, ready: false })),
      test: vi.fn(async () => ({ ok: true, provider: "sherpa", loadMs: 12 })),
    };
    const caller = createCaller(localCaptionRouter, { localCaption } as never);

    await expect(caller.status()).resolves.toEqual(state);
    await expect(caller.install()).resolves.toEqual(state);
    await expect(caller.test()).resolves.toMatchObject({ ok: true, provider: "sherpa" });
    await expect(caller.uninstall()).resolves.toMatchObject({ installed: false });
    expect(localCaption.install).toHaveBeenCalledOnce();
    expect(localCaption.uninstall).toHaveBeenCalledOnce();
  });

  it("forwards the model key on install and uninstall", async () => {
    const state = { installed: true, ready: true, operation: "idle" };
    const localCaption = {
      status: vi.fn(() => state),
      install: vi.fn(async () => state),
      uninstall: vi.fn(async () => state),
      test: vi.fn(async () => ({ ok: true, provider: "sherpa", loadMs: 1 })),
    };
    const caller = createCaller(localCaptionRouter, { localCaption } as never);

    await caller.install({ model: "offline-final" });
    expect(localCaption.install).toHaveBeenCalledWith({ model: "offline-final" });
    await caller.uninstall({ model: "silero-vad" });
    expect(localCaption.uninstall).toHaveBeenCalledWith({ model: "silero-vad" });
  });

  it("rejects unknown model keys at the schema layer", async () => {
    const localCaption = {
      status: vi.fn(() => ({ installed: false, ready: false, operation: "idle" })),
      install: vi.fn(async () => ({ installed: false })),
      uninstall: vi.fn(async () => ({ installed: false })),
      test: vi.fn(async () => ({ ok: true })),
    };
    const caller = createCaller(localCaptionRouter, { localCaption } as never);
    await expect(caller.install({ model: "gpt" })).rejects.toThrow();
    expect(localCaption.install).not.toHaveBeenCalled();
  });

  it("delegates the offline smoke test", async () => {
    const localCaption = {
      status: vi.fn(() => ({ installed: true, ready: true, operation: "idle" })),
      install: vi.fn(async () => ({ installed: true })),
      uninstall: vi.fn(async () => ({ installed: false })),
      testOffline: vi.fn(async () => ({
        ok: true,
        provider: "sherpa-onnx-fire-red-asr-large-int8",
        loadMs: 42,
        segments: 0,
      })),
    };
    const caller = createCaller(localCaptionRouter, { localCaption } as never);

    await expect(caller.testOffline()).resolves.toEqual({
      ok: true,
      provider: "sherpa-onnx-fire-red-asr-large-int8",
      loadMs: 42,
      segments: 0,
    });
    expect(localCaption.testOffline).toHaveBeenCalledOnce();
  });
});
