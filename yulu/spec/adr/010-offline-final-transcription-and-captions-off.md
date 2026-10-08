# ADR-010: Offline final transcription and captions off by default

Status: Accepted

Restores the ADR-006 two-stage quality split inside the `local` engine of
ADR-007. ADR-007's explicit engine selection and visible-failure policy remain
unchanged.

## Context

The `local` engine used the streaming Paraformer replay for final transcripts.
Streaming models trade accuracy for latency, so final transcripts of long
meetings lost quality while CPU sat mostly idle after recording stopped.
Meanwhile realtime captions and the recording status window are opt-in UX, and
dictation needs speed rather than highest accuracy.

FireRedASR (sherpa-onnx `fire-red-asr-large` INT8) runs offline, one-shot VAD +
batch decode at roughly 0.32–0.35 RTF with 4 threads on Apple silicon, with no
punctuation differences needing special handling.

## Decision

Within the `local` engine, split transcription into two quality tiers:

- Streaming tier (fast): the existing Paraformer streaming replay stays the
  engine for realtime captions and dictation. Dictation requests send
  `tier: "fast"`.
- Final tier: final transcription of a completed recording runs a one-shot
  `sherpa_offline_worker.py` (FireRedASR INT8 + silero-vad) over source-separated
  16 kHz mic/system WAVs, merges cross-source duplicates, and commits the
  durable transcript. Selected via `transcription.local.final_model`
  (`fire-red` default; `paraformer-replay` keeps the old single-stage path).

Engine behavior follows ADR-007 unchanged: no automatic engine fallback. If
`fire-red` is selected but its assets are missing, the task fails visibly with
an actionable message instead of silently downgrading to the replay path.

Captions and the recording status window default to off
(`transcription.captions.realtime_enabled` / `status_window_enabled`).

Offline worker resource policy: threads default to `min(4, cores - 2)` with a
floor of 2, overridable via `transcription.local.offline_threads`;
`OMP_NUM_THREADS=1`; the process is killed after a 90-minute timeout. The
offline model (~1.5 GB) and silero-vad are user-managed from Settings,
checksum-pinned like the streaming runtime, and downloadable independently of it.

## Consequences

- Final transcripts regain offline high quality without a cloud dependency or
  a new engine selection; realtime paths keep their low latency.
- Streaming runtime availability no longer depends on the 1.5 GB offline model,
  and vice versa; each asset installs and uninstalls independently.
- First automatic transcription after recording on a default config requires
  the offline model to be installed; until then tasks fail with instructions.
- Dictation latency is unchanged; its transcripts stay on the fast tier.
