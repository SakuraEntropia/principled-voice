from __future__ import annotations

import numpy as np
from scipy.signal import correlate, find_peaks


def analyze(audio: np.ndarray, sr: int, conditioning_f0: np.ndarray | None = None) -> dict:
    """Independent autocorrelation estimator; uncertain/unvoiced frames are excluded.

    No claims of ground-truth pitch, emotion, intelligibility or speaker identity.
    Conditioning alignment uses normalized time and is approximate near padding.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("audio must be non-empty finite mono waveform")
    win, hop = int(0.05 * sr), int(0.01 * sr)
    f0, confidence, rms, times = [], [], [], []
    for start in range(0, max(0, len(x) - win + 1), hop):
        frame = x[start:start + win]
        power = np.sqrt(np.mean(frame * frame))
        rms.append(float(power))
        times.append((start + win / 2) / sr)
        frame = (frame - frame.mean()) * np.hanning(win)
        corr = correlate(frame, frame, mode="full", method="fft")[win - 1:]
        lo, hi = int(sr / 800), min(int(sr / 50), len(corr) - 1)
        corr /= max(corr[0], 1e-15)
        peaks, _ = find_peaks(corr[lo:hi])
        if power < 0.008 or len(peaks) == 0:
            f0.append(float("nan")); confidence.append(0.0)
            continue
        candidates = peaks + lo
        best = candidates[np.argmax(corr[candidates])]
        conf = float(corr[best])
        offset = 0.0
        # Quadratic refinement, keeping the signed curvature.
        curvature = corr[best - 1] - 2 * corr[best] + corr[best + 1]
        if abs(curvature) > 1e-12:
            offset = float(np.clip(0.5 * (corr[best - 1] - corr[best + 1]) / curvature, -0.5, 0.5))
        f0.append(float(sr / (best + offset)) if conf > 0.6 else float("nan"))
        confidence.append(conf)
    f0 = np.asarray(f0)
    voiced = np.isfinite(f0)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freq = np.fft.rfftfreq(len(x), 1 / sr)
    result = {
        "duration_s": len(x) / sr, "sample_count": len(x), "finite": True,
        "rms_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12))),
        "peak": float(np.max(np.abs(x))), "over_full_scale_samples": int(np.sum(np.abs(x) > 1)),
        "spectral_centroid_hz": float(np.sum(freq * spectrum) / max(np.sum(spectrum), 1e-12)),
        "f0_estimator": "windowed_autocorrelation_50ms_10ms_confidence_gt_0.6",
        "f0_median_hz": float(np.median(f0[voiced])) if voiced.any() else None,
        "voiced_frame_fraction": float(voiced.mean()) if len(voiced) else 0,
        "f0_hz": [float(v) if np.isfinite(v) else None for v in f0],
        "times_s": times, "confidence": confidence,
        "not_measured": ["speaker_similarity", "emotion_accuracy", "content_preservation",
                         "naturalness", "ground_truth_phoneme_duration_error"],
    }
    if conditioning_f0 is not None and voiced.any():
        requested = np.interp(np.array(times) / (len(x) / sr),
                              np.linspace(0, 1, len(conditioning_f0)), conditioning_f0)
        valid = voiced & (requested > 40)
        result["conditioning_tracking_mae_cents"] = float(np.mean(
            np.abs(1200 * np.log2(f0[valid] / requested[valid]))
        )) if valid.any() else None
        result["tracking_alignment"] = "approximate_normalized_time_not_ground_truth"
    return result
