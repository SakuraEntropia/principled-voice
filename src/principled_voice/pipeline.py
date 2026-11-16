from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .analysis import analyze
from .core.schema import Project
from .synthesis.kokoro import SAMPLE_RATE, KokoroBackend, sha256


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def render(project: Project, backend: KokoroBackend, output: Path) -> dict:
    # Each directory is a new take. Never overwrite an existing render or project.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "project.json", project.model_dump(mode="json"))
    start = time.perf_counter()
    try:
        result = backend.synthesize(project)
        elapsed = time.perf_counter() - start
        analysis = analyze(result.audio, SAMPLE_RATE, result.arrays["used_f0"])
        locks = []
        for lock in project.locks:
            measured = analysis["f0_median_hz"]
            violation = None if measured is None else abs(measured - lock.target)
            locks.append({**lock.model_dump(), "measured": measured,
                          "violation": violation,
                          "satisfied": violation is not None and violation <= lock.tolerance,
                          "scope": "median_voiced_waveform_f0_not_every_frame"})
        hard_pass = all(x["satisfied"] for x in locks if x["strength"] == "hard")
        # Float WAV preserves amplitude; no hidden clipping/normalization.
        sf.write(output / "voice.wav", result.audio, SAMPLE_RATE, subtype="FLOAT")
        np.savez_compressed(output / "conditioning.npz", **result.arrays)
        write_json(output / "alignment.json", result.metadata.pop("alignment"))
        write_json(output / "analysis.json", analysis)
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "kokoro", "misaki", "numpy", "scipy")}
        manifest = {
            "schema_version": 1, "status": "rendered" if hard_pass else "constraint_violation",
            "production_approved": False, "identity_status": "temporary_voice",
            "project_sha256": sha256(output / "project.json"),
            "waveform_sha256": sha256(output / "voice.wav"),
            "audio_samples_sha256": hashlib.sha256(result.audio.astype("<f4").tobytes()).hexdigest(),
            "conditioning_sha256": sha256(output / "conditioning.npz"),
            "elapsed_s_excluding_model_load": elapsed,
            "real_time_factor": elapsed / analysis["duration_s"],
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_semantics": "macOS lifetime process high-water mark; not GPU allocation",
            "platform": platform.platform(), "python": platform.python_version(),
            "versions": versions, "locks": locks, **result.metadata,
            "shot": {"scene": project.scene, "shot": project.shot, "take": project.take,
                     "fps": project.fps, "start_frame": project.start_frame,
                     "duration_frames": analysis["duration_s"] * project.fps},
        }
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
