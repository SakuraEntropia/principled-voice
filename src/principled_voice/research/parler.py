"""A small, frozen real-TTS cache intervention pilot, not an emotion benchmark.

Self-cache prefixes are transplanted at identical absolute positions. Donor and source
have the same linguistic prompt; style description only changes cross conditioning.
The pilot deliberately leaves source cross-attention unchanged, so it is not a full
reproduction of arXiv:2605.27376's combined style replacement method.
"""
from __future__ import annotations

import hashlib
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ..analysis import analyze
from ..pipeline import write_json
from ..synthesis.kokoro import sha256
from .kv import CacheBlock, Intervention, patch_cache, prefix_window_mask

PARLER_REVISION = "fe1bd939bb05464d39a76e20cf8a35d4e6885571"
TEXT = "The light is still here."
SOURCE_STYLE = ("A female speaker delivers a slightly expressive and animated speech with a moderate "
                "speed and pitch. The recording is of very high quality, with the speaker's voice "
                "sounding clear and very close up.")
TARGET_STYLE = "A female speaker speaks excitedly with a high pitch and very clear audio."


def run_experiment(model_dir: Path, output: Path, max_tokens: int = 350,
                   sampling: bool = False) -> dict:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    if not 100 <= max_tokens <= 500:
        raise ValueError("pilot generation budget must be 100–500 tokens")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_dir = Path(model_dir)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    start = time.perf_counter()
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, attn_implementation="eager"
    ).eval().to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    layers = model.decoder.model.decoder.layers
    for p in model.parameters():
        p.requires_grad_(False)
    prompt = tokenizer(TEXT, return_tensors="pt")
    prefix = prompt.input_ids.shape[1] + 8
    switch = prefix + 32
    text_hash = hashlib.sha256(prompt.input_ids.numpy().tobytes()).hexdigest()
    donor: dict[int, CacheBlock] = {}
    source: dict[int, CacheBlock] = {}
    donor_activation, source_activation = {}, {}
    report = {"model": "parler-tts/parler-tts-tiny-v1", "revision": PARLER_REVISION,
              "device": "cpu", "text": TEXT, "source_description": SOURCE_STYLE,
              "target_description": TARGET_STYLE, "seed": 42,
              "decoding": "model_default_sampling" if sampling else "greedy",
              "prefix_positions": prefix, "intervene_at_cache_position": switch,
              "max_new_tokens": max_tokens, "layers": len(layers),
              "positional_encoding": "sinusoidal", "cross_attention": "source_unchanged",
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "model_sha256": sha256(model_dir / "model.safetensors"),
              "load_elapsed_s": time.perf_counter() - start, "cases": {},
              "claim": "engineering pilot; no claim of emotion/identity preservation",
              "scope_limits": ["one text, one seed", "no identity lock evaluator",
                               "not a faithful mini-model paper reproduction", "no listening panel",
                               "fixed token budget may truncate", "donor audio token histories differ"]}

    def block(k, v, layer):
        return CacheBlock(k[..., :prefix, :].detach().clone(),
                          v[..., :prefix, :].detach().clone(), PARLER_REVISION, layer,
                          tuple(range(prefix)), text_hash, "sinusoidal")

    def capture_hook(store, activations, layer):
        def hook(module, args, kwargs, result):
            cache = kwargs.get("past_key_value")
            if cache is None:
                return
            cache = cache.self_attention_cache
            if len(cache.key_cache) <= layer:
                return
            k, v = cache.key_cache[layer], cache.value_cache[layer]
            if k.shape[-2] >= prefix and layer not in store:
                store[layer] = block(k, v, layer)
            # Initial query contains the text prefix and first decoder position.
            if layer not in activations:
                activations[layer] = result[0].detach().mean(dim=1, keepdim=True).clone()
        return hook

    @torch.inference_mode()
    def generate(name, description, edit=None, selected_layers=None, window=None, activation=False,
                 capture=None, activations=None):
        handles, events, originals = [], [], {}
        style = tokenizer(description, return_tensors="pt")
        for i, layer in enumerate(layers):
            if capture is not None:
                handles.append(layer.self_attn.register_forward_hook(
                    capture_hook(capture, activations, i), with_kwargs=True))
            if edit is not None and (selected_layers is None or i in selected_layers):
                def prehook(module, args, kwargs, idx=i):
                    cache = kwargs.get("past_key_value")
                    if cache is None or len(cache.self_attention_cache.key_cache) <= idx:
                        return
                    self_cache = cache.self_attention_cache
                    k, v = self_cache.key_cache[idx], self_cache.value_cache[idx]
                    position = k.shape[-2]
                    if position < switch:
                        return
                    if idx not in originals:
                        originals[idx] = block(k, v, idx)
                    # Fixed baseline prefix, so continuous application cannot accumulate.
                    edited = patch_cache(originals[idx], donor[idx], edit)
                    self_cache.key_cache[idx][..., :prefix, :] = edited.k
                    self_cache.value_cache[idx][..., :prefix, :] = edited.v
                    if idx not in events:
                        events.append(idx)
                    if window is not None:
                        qlen = kwargs["hidden_states"].shape[1]
                        visible = prefix_window_mask(torch.arange(position, position + qlen),
                                                     torch.arange(position + qlen), prefix, window)
                        additive = torch.zeros_like(visible, dtype=k.dtype).masked_fill(
                            ~visible, torch.finfo(k.dtype).min)[None, None]
                        old = kwargs.get("attention_mask")
                        kwargs["attention_mask"] = additive if old is None else old + additive
                    return args, kwargs
                handles.append(layer.self_attn.register_forward_pre_hook(prehook, with_kwargs=True))
            if activation and i == len(layers) // 2:
                def activation_hook(module, args, kwargs, result, idx=i):
                    cache = kwargs.get("past_key_value")
                    if cache is None or cache.self_attention_cache.key_cache[idx].shape[-2] < switch:
                        return
                    delta = donor_activation[idx] - source_activation[idx]
                    if idx not in events:
                        events.append(idx)
                    return (result[0] + 0.1 * delta, *result[1:])
                handles.append(layer.self_attn.register_forward_hook(activation_hook, with_kwargs=True))
        torch.manual_seed(42)
        t0 = time.perf_counter()
        try:
            audio = model.generate(input_ids=style.input_ids, attention_mask=style.attention_mask,
                                   prompt_input_ids=prompt.input_ids,
                                   prompt_attention_mask=prompt.attention_mask,
                                   do_sample=sampling, max_new_tokens=max_tokens,
                                   min_new_tokens=80, use_cache=True).squeeze().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        elapsed = time.perf_counter() - t0
        sf.write(output / f"{name}.wav", audio, model.config.sampling_rate, subtype="FLOAT")
        stats = analyze(audio, model.config.sampling_rate)
        write_json(output / f"{name}_analysis.json", stats)
        report["cases"][name] = {"duration_s": stats["duration_s"], "rms_dbfs": stats["rms_dbfs"],
                                  "f0_median_hz": stats["f0_median_hz"],
                                  "voiced_frame_fraction": stats["voiced_frame_fraction"],
                                  "elapsed_s": elapsed, "patched_layers": events,
                                  "wav_sha256": sha256(output / f"{name}.wav")}
        write_json(output / "results.json", report)
        return audio

    try:
        generate("native_target", TARGET_STYLE, capture=donor, activations=donor_activation)
        baseline = generate("baseline", SOURCE_STYLE, capture=source, activations=source_activation)
        if set(donor) != set(range(len(layers))):
            raise RuntimeError("donor generation too short to capture every layer")
        noop = generate("noop", SOURCE_STYLE, Intervention(alpha=0))
        report["noop_sample_exact"] = bool(np.array_equal(baseline, noop))
        report["noop_max_abs_error"] = float(np.max(np.abs(baseline - noop))) if baseline.shape == noop.shape else None
        if not report["noop_sample_exact"]:
            raise RuntimeError("no-op changed audio; intervention experiment invalid")
        for name, edit, selected, window in [
            ("k_only", Intervention("k", 1), None, None),
            ("v_only", Intervention("v", 1), None, None),
            ("joint_kv", Intervention("kv", 0.5), None, None),
            ("cache_swap", Intervention("kv", 1, relative_norm_cap=1e6), None, None),
            ("low_rank", Intervention("kv", 0.5, rank=4), None, None),
            ("layer_head", Intervention("kv", 0.5, heads=(0,)), (len(layers) // 2,), None),
            ("swap_window", Intervention("kv", 1, relative_norm_cap=1e6), None, 32),
        ]:
            generate(name, SOURCE_STYLE, edit, selected, window)
        generate("activation", SOURCE_STYLE, activation=True)
        acoustic_valid = all(report["cases"][name]["rms_dbfs"] > -40 and
                             report["cases"][name]["voiced_frame_fraction"] > 0.1
                             for name in ("baseline", "native_target"))
        report["acoustic_signal_gate_passed"] = acoustic_valid
        report["status"] = ("pilot_completed_not_emotion_validated" if acoustic_valid
                            else "invalid_baseline_signal_no_emotion_conclusion")
        report["cache_shapes"] = {str(i): list(b.k.shape) for i, b in donor.items()}
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["total_elapsed_s"] = time.perf_counter() - start
        write_json(output / "results.json", report)
        return {"status": report["status"], "output": str(output),
                "noop_exact": report["noop_sample_exact"], "cases": len(report["cases"])}
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output / "results.json", report)
        raise
