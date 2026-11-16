"""Fetch pinned public models; never uploads character assets or recordings."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from principled_voice.research.parler import PARLER_REVISION
from principled_voice.synthesis.kokoro import KOKORO_REVISION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("models"))
    p.add_argument("--with-kv", action="store_true")
    p.add_argument("--with-asr", action="store_true")
    args = p.parse_args()
    snapshot_download("hexgrad/Kokoro-82M", revision=KOKORO_REVISION,
                      allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "README.md", "VOICES.md"],
                      local_dir=str(args.root / "kokoro"))
    if args.with_kv:
        snapshot_download("parler-tts/parler-tts-tiny-v1", revision=PARLER_REVISION,
                          ignore_patterns=["*.bin", "*.pt", "*.h5", "*.msgpack"],
                          local_dir=str(args.root / "parler"))
    if args.with_asr:
        snapshot_download("Systran/faster-whisper-tiny.en",
                          revision="0d3d19a32d3338f10357c0889762bd8d64bbdeba",
                          local_dir=str(args.root / "whisper-tiny.en"))


if __name__ == "__main__":
    main()
