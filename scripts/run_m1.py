"""Run actual neural controls and finite-difference observations; no mock backend."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from principled_voice.core.schema import Project
from principled_voice.pipeline import render, write_json
from principled_voice.synthesis.kokoro import KokoroBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/kokoro"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.perf_counter()
    backend = KokoroBackend(args.model_dir, "cpu")
    model_load_s = time.perf_counter() - start
    base = Project(text="This is a temporary voice for a technical test. The light is still here.")
    cases = {
        "baseline": {}, "repeat": {}, "pitch_up": {"pitch_semitones": 3},
        "pitch_down": {"pitch_semitones": -3}, "faster": {"speed": 1.2},
        "slower": {"speed": 0.8}, "n_up": {"n_offset": 0.3}, "n_down": {"n_offset": -0.3},
        "aroused": {"performance": {"arousal": 0.7, "inhibition": 0}},
        "inhibited": {"performance": {"arousal": 0.7, "inhibition": 1}},
        "pitch_curve": {"pitch_curve": {"keys": [{"time": 0, "value": -2},
                                                    {"time": 1, "value": 3}]}},
        "absolute_f0": {"f0_hz": {"keys": [{"time": 0, "value": 200}]}},
    }
    results = {}
    for name, controls in cases.items():
        project = Project.model_validate({**base.model_dump(), "take": name, "controls": controls})
        manifest = render(project, backend, args.out / name)
        import json
        analysis = json.loads((args.out / name / "analysis.json").read_text())
        results[name] = {k: analysis[k] for k in
                         ("duration_s", "f0_median_hz", "rms_dbfs", "spectral_centroid_hz",
                          "conditioning_tracking_mae_cents", "peak")}
        results[name]["waveform_sha256"] = manifest["waveform_sha256"]
        results[name]["audio_samples_sha256"] = manifest["audio_samples_sha256"]
        results[name]["rtf"] = manifest["real_time_factor"]
    # Explicit native token-duration override: preserve original generated alignment.
    durations = np.load(args.out / "baseline/conditioning.npz")["used_durations"].tolist()
    durations[len(durations) // 2] *= 2
    edited = Project.model_validate({**base.model_dump(), "take": "token_duration",
                                    "controls": {"token_durations": durations}})
    token_m = render(edited, backend, args.out / "token_duration")
    # A strict hard lock is evaluated from output waveform, not assumed satisfied.
    locked = Project.model_validate({**base.model_dump(), "take": "strict_lock",
                                    "locks": [{"parameter": "f0_hz", "target": 200,
                                               "tolerance": 0.01, "strength": "hard"}]})
    lock_m = render(locked, backend, args.out / "strict_lock")
    # Frozen upstream equivalence at neutral controls, no model reinitialization.
    ps = backend.phonemize(base)
    torch.manual_seed(base.seed)
    with torch.inference_mode():
        upstream = backend.model(ps, backend.voice[len(ps) - 1], return_output=True).audio.numpy()
    import soundfile as sf
    baseline_audio, _ = sf.read(args.out / "baseline/voice.wav", dtype="float32")
    upstream_error = float(np.max(np.abs(upstream - baseline_audio)))
    observables = ["f0_median_hz", "duration_s", "rms_dbfs", "spectral_centroid_hz"]
    pairs = [("pitch_semitones", "pitch_up", "pitch_down", 6),
             ("speed", "faster", "slower", 0.4), ("n_offset", "n_up", "n_down", 0.6)]
    jacobian = {control: {k: (results[up][k] - results[down][k]) / delta for k in observables}
                for control, up, down, delta in pairs}
    summary = {"status": "measured", "model_load_s": model_load_s, "cases": results,
               "repeat_exact": results["baseline"]["audio_samples_sha256"] == results["repeat"]["audio_samples_sha256"],
               "neutral_matches_upstream_max_abs_error": upstream_error,
               "token_duration_expected_samples": token_m["expected_samples"],
               "strict_lock": {"status": lock_m["status"], "diagnostics": lock_m["locks"]},
               "finite_difference_jacobian": jacobian,
               "jacobian_scope": "one utterance, one seed, symmetric finite differences; not a universal Jacobian",
               "identity_emotion_content_naturalness": "not measured",
               "total_elapsed_s": time.perf_counter() - start}
    write_json(args.out / "results.json", summary)
    assert summary["repeat_exact"], "fixed-seed repeat changed"
    assert upstream_error < 1e-5, "neutral adapter does not match upstream inference"
    assert results["pitch_up"]["f0_median_hz"] > results["pitch_down"]["f0_median_hz"]
    assert results["faster"]["duration_s"] < results["slower"]["duration_s"]
    assert summary["strict_lock"]["status"] == "constraint_violation", "strict lock was silently accepted"
    print(f"M1 experiments saved: {args.out}")


if __name__ == "__main__":
    main()
