import { z } from "zod";
import { publicProcedure, router } from "../trpc.js";

const modelSchema = z.enum(["streaming", "offline-final", "silero-vad"]).optional();

function manager(ctx: { localCaption?: import("../localCaptionManager.js").LocalCaptionManager }) {
  if (!ctx.localCaption) throw new Error("本地实时转录管理器不可用");
  return ctx.localCaption;
}

export const localCaptionRouter = router({
  status: publicProcedure.query(({ ctx }) => manager(ctx).status()),
  install: publicProcedure
    .input(z.object({ model: modelSchema }).optional())
    .mutation(async ({ ctx, input }) => await manager(ctx).install(input)),
  uninstall: publicProcedure
    .input(z.object({ model: modelSchema }).optional())
    .mutation(async ({ ctx, input }) => await manager(ctx).uninstall(input)),
  test: publicProcedure.mutation(async ({ ctx }) => await manager(ctx).test()),
  testOffline: publicProcedure.mutation(async ({ ctx }) => await manager(ctx).testOffline()),
});
