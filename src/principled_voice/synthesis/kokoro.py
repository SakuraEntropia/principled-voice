# Inference alignment adapted from hexgrad/kokoro model.py (Apache-2.0).
# Original: hexgrad/kokoro. Modifications: explicit token duration, F0/N controls and traces.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from ..core.coupling import evaluate_performance
from ..core.schema import Project

KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SAMPLE_RATE = 24000
DURATION_HOP = 600
MAX_FRAMES = 1200  # 30s; fail before dense alignment allocation.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class SynthesisResult:
    audio: np.ndarray
    arrays: dict[str, np.ndarray]
    metadata: dict


class KokoroBackend:
    capabilities: ClassVar[dict[str, bool]] = {"native_f0": True, "native_token_duration": True,
                    "native_n_condition": True, "kv_cache": False,
                    "phoneme_ground_truth": False, "trained_emotion": False}

    def __init__(self, model_dir: Path, device: str = "cpu"):
        from kokoro import KModel

        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps or cuda")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable in this process; no silent CPU fallback")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.root = Path(model_dir).resolve()
        self.device = torch.device(device)
        self._frontend = None
        self.model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(self.root / "config.json"),
                            model=str(self.root / "kokoro-v1_0.pth"),
                            disable_complex=(device == "mps")).to(device).eval()
        self.voice = torch.load(self.root / "voices/af_heart.pt", weights_only=True,
                                map_location="cpu")
        self.hashes = {name: sha256(self.root / name) for name in
                       ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt")}

    def phonemize(self, project: Project) -> str:
        if project.phonemes:
            ps = project.phonemes
        else:
            from .frontend import EnglishFrontend

            if self._frontend is None:
                self._frontend = EnglishFrontend()
            ps = self._frontend(project.text)
        if not ps:
            raise ValueError("G2P returned empty phonemes")
        unknown = sorted(set(ps) - self.model.vocab.keys())
        if unknown:
            raise ValueError(f"unsupported phonemes: {unknown}; provide corrected phonemes")
        if len(ps) + 2 > self.model.context_length:
            raise ValueError("utterance exceeds backbone context; split explicitly")
        return ps

    @torch.inference_mode()
    def synthesize(self, project: Project) -> SynthesisResult:
        torch.manual_seed(project.seed)
        if self.device.type == "mps":
            torch.mps.manual_seed(project.seed)
        ps = self.phonemize(project)
        m = self.model
        c = project.controls
        ids = torch.tensor([[0, *[m.vocab[p] for p in ps], 0]], device=self.device)
        lengths = torch.tensor([ids.shape[1]], device=self.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        ref = self.voice[len(ps) - 1].to(self.device)
        if ref.shape != (1, 256):
            raise ValueError(f"unexpected voice embedding shape: {tuple(ref.shape)}")
        style = ref[:, 128:]
        encoded = m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
        dur_features = m.predictor.text_encoder(encoded, style, lengths, mask)
        hidden, _ = m.predictor.lstm(dur_features)
        float_dur = torch.sigmoid(m.predictor.duration_proj(hidden)).sum(-1).squeeze(0)
        generated_dur = torch.round(float_dur).clamp(min=1).long()
        durations = torch.round(float_dur / c.speed).clamp(min=1).long()
        if c.token_durations is not None:
            if len(c.token_durations) != ids.shape[1]:
                raise ValueError(f"duration count must be {ids.shape[1]} including BOS/EOS")
            durations = torch.tensor(c.token_durations, device=self.device)
        if int(durations.sum()) > MAX_FRAMES:
            raise ValueError("30 second synthesis budget exceeded; split the clip")
        index = torch.repeat_interleave(torch.arange(ids.shape[1], device=self.device), durations)
        alignment = torch.nn.functional.one_hot(index, ids.shape[1]).T.float().unsqueeze(0)
        prosody = dur_features.transpose(-1, -2) @ alignment
        f0, n = m.predictor.F0Ntrain(prosody, style)
        generated_f0, generated_n = f0.clone(), n.clone()
        trace = evaluate_performance(c.performance.arousal, c.performance.inhibition)
        dist = trace["prosody_distribution"]
        voiced = f0 > 10
        if voiced.any():
            log_f0 = torch.log2(f0.clamp(min=1))
            center = log_f0[voiced].mean()
            # Distribution variance is modulated; no random performance resampling.
            spread = 1 + 0.3 * (dist["pitch_std_st"] - 0.2) / 0.5
            if spread != 1:
                f0 = torch.where(voiced, 2 ** (center + (log_f0 - center) * spread), f0)
        st = np.full(f0.shape[-1], c.pitch_semitones + dist["pitch_mean_st"])
        if c.pitch_curve:
            st += c.pitch_curve.sample(len(st))
        f0 = f0 * torch.as_tensor(2 ** (st / 12), device=self.device, dtype=f0.dtype)
        absolute = c.f0_hz.sample(f0.shape[-1]) if c.f0_hz else None
        for lock in project.locks:
            if lock.parameter == "f0_hz":
                absolute = np.full(f0.shape[-1], lock.target)
        if absolute is not None:
            requested = torch.as_tensor(absolute, device=self.device, dtype=f0.dtype)[None, :]
            f0 = torch.where(voiced, requested, f0)
        f0_edit_active = (c.pitch_semitones != 0 or c.pitch_curve is not None
                          or absolute is not None or dist["pitch_mean_st"] != 0)
        if f0_edit_active:
            f0 = torch.where(voiced, f0.clamp(40, 800), f0)
        n = n + c.n_offset + dist["n_mean"]
        if c.n_curve:
            n = n + torch.as_tensor(c.n_curve.sample(n.shape[-1]), device=self.device,
                                    dtype=n.dtype)[None, :]
        asr = m.text_encoder(ids, lengths, mask) @ alignment
        audio = m.decoder(asr, f0, n, ref[:, :128]).squeeze().float().cpu().numpy()
        audio *= 10 ** (c.output_gain_db / 20)
        arrays = {"generated_durations": generated_dur.cpu().numpy(),
                  "used_durations": durations.cpu().numpy(),
                  "generated_f0": generated_f0.cpu().numpy().squeeze(),
                  "used_f0": f0.cpu().numpy().squeeze(),
                  "generated_n": generated_n.cpu().numpy().squeeze(),
                  "used_n": n.cpu().numpy().squeeze()}
        boundaries = np.concatenate([[0], np.cumsum(arrays["used_durations"])])
        tokens = ["<bos>", *ps, "<eos>"]
        alignment_json = [{"token": token, "start_s": float(boundaries[i] * 0.025),
                           "end_s": float(boundaries[i + 1] * 0.025),
                           "provenance": "model_alignment_not_forced_alignment"}
                          for i, token in enumerate(tokens)]
        return SynthesisResult(audio, arrays, {
            "model": "hexgrad/Kokoro-82M", "revision": KOKORO_REVISION,
            "model_files_sha256": self.hashes, "device": str(self.device),
            "sample_rate": SAMPLE_RATE, "voice": "Temporary Voice / af_heart",
            "capabilities": self.capabilities, "phonemes": ps, "alignment": alignment_json,
            "coupling": trace, "duration_hop_samples": DURATION_HOP,
            "expected_samples": int(durations.sum()) * DURATION_HOP,
            "post_effects": {"output_gain_db": c.output_gain_db},
        })


def load_project(path: Path) -> Project:
    return Project.model_validate(json.loads(Path(path).read_text()))
