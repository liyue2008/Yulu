#!/usr/bin/env python3
"""One-shot offline transcription worker for Yulu's local final transcripts.

Spawned per task by the Host: loads the FireRedASR offline model plus silero
VAD, segments each pre-resampled 16 kHz source WAV, decodes the segments in
batches, prints newline-delimited progress events, prints one final result
event, and exits. It never writes transcript artifacts; the Host stays the
owner of transcription state and persistence. Audio never leaves the machine.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import wave
from array import array
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
SOURCES = ("mic", "system")
MODEL_FILES = ("tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx")
VAD_WINDOW = 512
SEGMENT_PAD_SEC = 0.2  # context padding around each VAD segment to avoid clipped words


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")), flush=True)


def _read_wav_pcm16(path: Path) -> array:
    """Read a 16 kHz mono signed-16-bit WAV into an array of raw samples."""
    try:
        with wave.open(str(path), "rb") as source:
            if source.getframerate() != SAMPLE_RATE:
                raise ValueError(f"{path.name}: 需要 {SAMPLE_RATE} Hz WAV，实际 {source.getframerate()} Hz")
            if source.getnchannels() != 1:
                raise ValueError(f"{path.name}: 需要单声道 WAV，实际 {source.getnchannels()} 声道")
            if source.getsampwidth() != 2:
                raise ValueError(f"{path.name}: 需要 16-bit PCM WAV，实际 {source.getsampwidth() * 8}-bit")
            raw = source.readframes(source.getnframes())
    except wave.Error as exc:
        raise ValueError(f"{path.name}: 无法读取 WAV: {exc}") from exc
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _as_floats(samples: array) -> list[float]:
    return [sample / 32768.0 for sample in samples]


class OfflineTranscriber:
    def __init__(
        self,
        model_dir: Path,
        vad_model: Path,
        *,
        threads: int = 4,
        batch_size: int = 8,
        vad_threshold: float = 0.5,
        vad_min_silence_sec: float = 0.7,
        vad_min_speech_sec: float = 0.25,
        vad_max_speech_sec: float = 25.0,
    ) -> None:
        import sherpa_onnx  # pyright: ignore[reportMissingImports]

        missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"离线模型不完整: {', '.join(missing)}")
        if not vad_model.is_file():
            raise FileNotFoundError(f"VAD 模型不存在: {vad_model.name}")
        self.sherpa = sherpa_onnx
        self.batch_size = max(1, batch_size)
        self.vad_max_speech_sec = vad_max_speech_sec
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=max(1, threads),
            decoding_method="greedy_search",
            provider="cpu",
        )
        vad_config = sherpa_onnx.SileroVadModelConfig(
            model=str(vad_model),
            threshold=vad_threshold,
            min_silence_duration=vad_min_silence_sec,
            min_speech_duration=vad_min_speech_sec,
            max_speech_duration=vad_max_speech_sec,
            window_size=VAD_WINDOW,
        )
        self.vad_config = sherpa_onnx.VadModelConfig(silero_vad=vad_config)

    def segments(self, samples: array) -> list[tuple[int, int]]:
        """Detect (start, end) speech spans, in samples, for one source."""
        detector = self.sherpa.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=60)
        window = VAD_WINDOW
        for offset in range(0, len(samples), window):
            chunk = samples[offset:offset + window]
            if len(chunk) < window:
                chunk = chunk + array("h", bytes(2 * (window - len(chunk))))
            detector.accept_waveform(_as_floats(chunk))
        detector.flush()
        spans: list[tuple[int, int]] = []
        while not detector.empty():
            segment = detector.front
            start = int(segment.start)
            end = start + len(segment.samples)
            spans.append((start, end))
            detector.pop()
        detector.reset()
        return self._merge_adjacent(spans)

    def _merge_adjacent(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge spans separated by sub-0.2s gaps while respecting max duration."""
        limit = int(self.vad_max_speech_sec * SAMPLE_RATE)
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged and start - merged[-1][1] < int(0.2 * SAMPLE_RATE) and end - merged[-1][0] <= limit:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def transcribe_source(self, samples: array, label: str) -> list[dict[str, Any]]:
        spans = self.segments(samples)
        total = len(spans)
        _emit("progress", phase="vad", source=label, segments=total,
              message=f"{label}: 检测到 {total} 段语音" if total else f"{label}: 未检测到语音")
        if not total:
            return []
        pad = int(SEGMENT_PAD_SEC * SAMPLE_RATE)
        # Longest-first batching keeps decode batches balanced; timestamps stay on the span.
        ordered = sorted(range(total), key=lambda index: (spans[index][1] - spans[index][0]), reverse=True)
        texts: dict[int, str] = {}
        for batch_start in range(0, total, self.batch_size):
            batch = ordered[batch_start:batch_start + self.batch_size]
            streams = []
            for index in batch:
                start, end = spans[index]
                lo = max(0, start - pad)
                hi = min(len(samples), end + pad)
                stream = self.recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, _as_floats(samples[lo:hi]))
                streams.append(stream)
            self.recognizer.decode_streams(streams)
            for index, stream in zip(batch, streams, strict=True):
                texts[index] = str(stream.result.text).strip()
            done = min(batch_start + self.batch_size, total)
            _emit("progress", phase="decode", percent=int(done * 100 / total),
                  message=f"{label}: 已解码 {done}/{total} 段")
        return [
            {
                "startMs": round(spans[index][0] / SAMPLE_RATE * 1000),
                "endMs": round(spans[index][1] / SAMPLE_RATE * 1000),
                "text": texts.get(index, ""),
            }
            for index in range(total)
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Yulu offline final-transcription worker")
    parser.add_argument("--runtime-pack", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--wav", action="append", default=[], metavar="SOURCE=PATH",
                        help="16 kHz mono WAV per source, e.g. mic=/path or system=/path")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default="zh", choices=("zh", "en", "auto"))
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-silence", type=float, default=0.7)
    parser.add_argument("--vad-min-speech", type=float, default=0.25)
    parser.add_argument("--vad-max-speech", type=float, default=25.0)
    args = parser.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with contextlib.suppress(OSError):
        os.nice(10)

    if args.language == "ja":
        print(json.dumps({"fatal": "本地离线模型仅支持中英文；日语请显式选择 xAI 引擎"},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        return 2

    try:
        trusted_script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(trusted_script_dir))
        from local_caption_runtime import verify_runtime_pack

        runtime_pack = args.runtime_pack.expanduser().resolve()
        verify_runtime_pack(runtime_pack)
        sys.path.insert(0, str(runtime_pack / "Contents/Resources/site-packages"))
        transcriber = OfflineTranscriber(
            args.model_dir.expanduser().resolve(),
            args.vad_model.expanduser().resolve(),
            threads=max(1, args.threads),
            batch_size=args.batch_size,
            vad_threshold=args.vad_threshold,
            vad_min_silence_sec=args.vad_min_silence,
            vad_min_speech_sec=args.vad_min_speech,
            vad_max_speech_sec=args.vad_max_speech,
        )
    except Exception as exc:
        print(json.dumps({"fatal": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2

    sources: dict[str, Path] = {}
    try:
        for entry in args.wav:
            label, _, value = entry.partition("=")
            if label not in SOURCES or not value:
                raise ValueError(f"--wav 需要 {SOURCES} 之一的 SOURCE=PATH 形式: {entry}")
            if label in sources:
                raise ValueError(f"重复的音频源: {label}")
            sources[label] = Path(value).expanduser()
        if not sources:
            raise ValueError("至少需要一个 --wav SOURCE=PATH")
        results: dict[str, list[dict[str, Any]]] = {}
        for label in SOURCES:
            path = sources.get(label)
            if path is None:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"音频文件不存在: {path}")
            results[label] = transcriber.transcribe_source(_read_wav_pcm16(path), label)
    except Exception as exc:
        print(json.dumps({"fatal": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2

    _emit("result", provider="sherpa-onnx-fire-red-asr-large-int8", sources=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
