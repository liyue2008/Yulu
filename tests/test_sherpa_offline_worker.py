import json
import sys
import types
import wave
from array import array
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "yulu" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import sherpa_offline_worker as worker  # noqa: E402


def write_wav(path: Path, samples: array, *, rate: int = 16_000, channels: int = 1, width: int = 2) -> None:
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(channels)
        sink.setsampwidth(width)
        sink.setframerate(rate)
        sink.writeframes(samples.tobytes())


def test_read_wav_pcm16_roundtrips_samples_and_enforces_16k_mono_16bit(tmp_path):
    samples = array("h", [-32768, -1, 0, 16384, 32767])
    good = tmp_path / "good.wav"
    write_wav(good, samples)
    assert worker._read_wav_pcm16(good) == samples

    wrong_rate = tmp_path / "rate.wav"
    write_wav(wrong_rate, samples, rate=8_000)
    with pytest.raises(ValueError, match="16000 Hz"):
        worker._read_wav_pcm16(wrong_rate)

    stereo = tmp_path / "stereo.wav"
    write_wav(stereo, samples, channels=2)
    with pytest.raises(ValueError, match="单声道"):
        worker._read_wav_pcm16(stereo)

    narrow = tmp_path / "narrow.wav"
    write_wav(narrow, samples, width=1)
    with pytest.raises(ValueError, match="16-bit"):
        worker._read_wav_pcm16(narrow)

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav at all")
    with pytest.raises(ValueError, match="无法读取 WAV"):
        worker._read_wav_pcm16(broken)


def make_transcriber(max_speech_sec: float = 25.0) -> worker.OfflineTranscriber:
    transcriber = worker.OfflineTranscriber.__new__(worker.OfflineTranscriber)
    transcriber.vad_max_speech_sec = max_speech_sec
    return transcriber


def test_merge_adjacent_bridges_sub_200_ms_gaps_only():
    transcriber = make_transcriber()
    # 0.19 s gap -> merged into one span.
    merged = transcriber._merge_adjacent([(1000, 5000), (5304, 9000)])
    assert merged == [(1000, 9000)]
    # 0.30 s gap -> kept apart.
    kept = transcriber._merge_adjacent([(1000, 5000), (9800, 12000)])
    assert kept == [(1000, 5000), (9800, 12000)]


def test_merge_adjacent_respects_max_speech_duration():
    transcriber = make_transcriber(max_speech_sec=1.0)
    limit = 16_000
    # Two ~0.94 s spans whose merge would exceed the 1 s limit stay separate.
    spans = [(0, limit - 800), (limit - 400, 2 * limit)]
    merged = transcriber._merge_adjacent(spans)
    assert merged == spans
    # Gaps under 200 ms inside the limit still merge.
    tight = transcriber._merge_adjacent([(0, limit - 800), (limit - 600, limit)])
    assert tight == [(0, limit)]


def fake_sherpa_module(span=(16_000, 32_000), text="你好，世界。"):
    sherpa = types.ModuleType("sherpa_onnx")

    def silero_config(**kwargs):
        return types.SimpleNamespace(**kwargs)

    def vad_config(silero_vad):
        return types.SimpleNamespace(silero_vad=silero_vad)

    class Front:
        def __init__(self, start, samples):
            self.start = start
            self.samples = samples

    class Detector:
        def __init__(self, _config, buffer_size_in_seconds=60):
            start, end = span
            self.pending = [Front(start, [0.0] * (end - start))]

        @property
        def front(self):
            return self.pending[0]

        def accept_waveform(self, samples):
            pass

        def flush(self):
            pass

        def empty(self):
            return not self.pending

        def pop(self):
            self.pending.pop(0)

        def reset(self):
            pass

    recognizer = types.SimpleNamespace()
    recognizer.create_stream = lambda: types.SimpleNamespace(
        accept_waveform=lambda rate, samples: None,
        result=types.SimpleNamespace(text=text),
    )
    recognizer.decode_streams = lambda streams: None
    sherpa.SileroVadModelConfig = silero_config  # type: ignore[attr-defined]
    sherpa.VadModelConfig = vad_config  # type: ignore[attr-defined]
    sherpa.VoiceActivityDetector = Detector  # type: ignore[attr-defined]
    sherpa.OfflineRecognizer = types.SimpleNamespace(from_fire_red_asr=lambda **kwargs: recognizer)  # type: ignore[attr-defined]
    return sherpa, recognizer


def stage_model_files(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    for name in worker.MODEL_FILES:
        (model_dir / name).write_bytes(b"")
    vad = tmp_path / "silero_vad.onnx"
    vad.write_bytes(b"")
    return model_dir, vad


def test_main_rejects_japanese_at_the_cli_layer(tmp_path):
    # argparse choices reject `ja` before any model load; the TS layer surfaces a
    # friendly message and never spawns the worker for Japanese.
    with pytest.raises(SystemExit) as excinfo:
        worker.main([
            "--runtime-pack", str(tmp_path / "pack.bundle"),
            "--model-dir", str(tmp_path / "model"),
            "--vad-model", str(tmp_path / "silero_vad.onnx"),
            "--wav", f"mic={tmp_path / 'a.wav'}",
            "--language", "ja",
        ])
    assert excinfo.value.code == 2


def test_main_end_to_end_emits_progress_and_result(monkeypatch, tmp_path, capsys):
    sherpa, _ = fake_sherpa_module()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", sherpa)
    monkeypatch.setattr(
        "local_caption_runtime.verify_runtime_pack",
        lambda pack: None,
    )
    model_dir, vad = stage_model_files(tmp_path)
    wav = tmp_path / "mic.wav"
    write_wav(wav, array("h", bytes(2 * 48_000)))  # 3 s of silence
    pack = tmp_path / "runtime.bundle"
    pack.mkdir()
    rc = worker.main([
        "--runtime-pack", str(pack),
        "--model-dir", str(model_dir),
        "--vad-model", str(vad),
        "--wav", f"mic={wav}",
        "--threads", "2",
        "--language", "zh",
    ])
    assert rc == 0
    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    phases = [(event["event"], event.get("phase")) for event in events]
    assert ("progress", "vad") in phases
    assert ("progress", "decode") in phases
    result = events[-1]
    assert result["event"] == "result"
    assert result["provider"] == "sherpa-onnx-fire-red-asr-large-int8"
    assert result["sources"]["mic"] == [{"startMs": 1000, "endMs": 2000, "text": "你好，世界。"}]


def test_main_rejects_unknown_wav_labels(monkeypatch, tmp_path, capsys):
    sherpa, _ = fake_sherpa_module()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", sherpa)
    monkeypatch.setattr(
        "local_caption_runtime.verify_runtime_pack",
        lambda pack: None,
    )
    model_dir, vad = stage_model_files(tmp_path)
    wav = tmp_path / "a.wav"
    write_wav(wav, array("h", bytes(2 * 16_000)))
    rc = worker.main([
        "--runtime-pack", str(tmp_path / "p.bundle"),
        "--model-dir", str(model_dir),
        "--vad-model", str(vad),
        "--wav", f"camera={wav}",
    ])
    assert rc == 2
    fatal = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert "SOURCE=PATH" in fatal["fatal"]


def test_main_requires_at_least_one_wav(monkeypatch, tmp_path, capsys):
    sherpa, _ = fake_sherpa_module()
    monkeypatch.setitem(sys.modules, "sherpa_onnx", sherpa)
    monkeypatch.setattr(
        "local_caption_runtime.verify_runtime_pack",
        lambda pack: None,
    )
    model_dir, vad = stage_model_files(tmp_path)
    rc = worker.main([
        "--runtime-pack", str(tmp_path / "p.bundle"),
        "--model-dir", str(model_dir),
        "--vad-model", str(vad),
    ])
    assert rc == 2
