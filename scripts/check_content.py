"""Independent ASR pilot. ASR agreement is not human intelligibility certification."""
import argparse
import json
import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

from principled_voice.synthesis.kokoro import sha256


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def wer(reference, hypothesis):
    a, b = words(reference), words(hypothesis)
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    return previous[-1] / max(len(a), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--m1", type=Path, required=True)
    p.add_argument("--kv", type=Path, required=True)
    args = p.parse_args()
    model = WhisperModel(str(args.model_dir), device="cpu", compute_type="int8", cpu_threads=4)
    cases = [(args.m1 / name / "voice.wav",
              "This is a temporary voice for a technical test. The light is still here.")
             for name in ("baseline", "pitch_up", "faster", "aroused")]
    cases += [(args.kv / f"{name}.wav", "The light is still here.")
              for name in ("baseline", "native_target", "k_only", "v_only", "cache_swap")]
    rows = []
    for path, reference in cases:
        if not path.exists():
            continue
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        factor = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // factor, sr // factor).astype(np.float32)
        # Array input avoids incompatible PyAV decode APIs across releases.
        segments, _ = model.transcribe(audio, language="en", beam_size=5,
                                       vad_filter=False, condition_on_previous_text=False)
        transcript = " ".join(s.text.strip() for s in segments)
        rows.append({"file": str(path), "reference": reference, "transcript": transcript,
                     "word_error_rate": wer(reference, transcript)})
    args.out.write_text(json.dumps({"evaluator": "Systran/faster-whisper-tiny.en",
                                    "revision": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
                                    "model_sha256": sha256(args.model_dir / "model.bin"),
                                    "scope": "single ASR model; not listening validation",
                                    "cases": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
