"""Download only public primary sources and record exact retrieval hashes/status.

Source snapshots live under a user-chosen audit folder, not in the distributable code.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SOURCES = {
    "fine_grained_style": "https://arxiv.org/html/2605.27376v1",
    "emotion_circuit": "https://arxiv.org/html/2610.05080v1",
    "ctrlspeech_paper": "https://arxiv.org/html/2608.08362v1",
    "ctrl_p": "https://arxiv.org/abs/2106.08352",
    "fastspeech2": "https://arxiv.org/abs/2006.04558",
    "fastspeech2_author_site": "https://speechresearch.github.io/fastspeech2/",
    "sparc": "https://arxiv.org/abs/2406.12998",
    "ddsp": "https://arxiv.org/abs/2001.04643",
    "nsf": "https://arxiv.org/abs/1904.12088",
    "hypernetworks": "https://arxiv.org/abs/1609.09106",
    "film": "https://arxiv.org/abs/1709.07871",
    "activation_steering": "https://arxiv.org/abs/2308.10248",
    "activation_patching": "https://arxiv.org/abs/2404.15255",
    "indextts2": "https://arxiv.org/abs/2506.21619",
    "parler_repo": "https://api.github.com/repos/huggingface/parler-tts",
    "kokoro_repo": "https://api.github.com/repos/hexgrad/kokoro",
    "ctrlspeech_repo": "https://api.github.com/repos/zszheng147/ctrlspeech",
    "sparc_repo": "https://api.github.com/repos/Berkeley-Speech-Group/Speech-Articulatory-Coding",
    "ctrlspeech_license": "https://raw.githubusercontent.com/zszheng147/ctrlspeech/main/LICENSE",
    "ctrlspeech_weights": "https://huggingface.co/zhisheng01/CtrlSpeech/raw/main/README.md",
    "indextts_license": "https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE",
    "kokoro_license": "https://raw.githubusercontent.com/hexgrad/kokoro/main/LICENSE",
    "kokoro_card": "https://huggingface.co/hexgrad/Kokoro-82M/raw/f3ff3571791e39611d31c381e3a41a3af07b4987/README.md",
    "kokoro_voices": "https://huggingface.co/hexgrad/Kokoro-82M/raw/f3ff3571791e39611d31c381e3a41a3af07b4987/VOICES.md",
    "parler_card": "https://huggingface.co/parler-tts/parler-tts-tiny-v1/raw/fe1bd939bb05464d39a76e20cf8a35d4e6885571/README.md",
    "parler_license": "https://raw.githubusercontent.com/huggingface/parler-tts/d108732cd57788ec86bc857d99a6cabd66663d68/LICENSE",
    "nsf_license": "https://raw.githubusercontent.com/nii-yamagishilab/project-NN-Pytorch-scripts/master/LICENSE",
    "ddsp_license": "https://raw.githubusercontent.com/magenta/ddsp/main/LICENSE",
    "pytorch_mps": "https://docs.pytorch.org/docs/stable/notes/mps.html",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    args.snapshots.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, url = item
        row = {"name": name, "url": url, "retrieved_utc": datetime.now(UTC).isoformat()}
        try:
            with urlopen(Request(url, headers={"User-Agent": "PrincipledVoice-SourceAudit/0.1"}), timeout=40) as r:
                data = r.read()
                row.update(status=r.status, resolved_url=r.url,
                           sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                (args.snapshots / f"{name}.source").write_bytes(data)
        except (OSError, HTTPError) as e:
            row.update(status="unverified", error=str(e))
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, SOURCES.items()))
    args.manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"verified": sum(r["status"] == 200 for r in rows), "total": len(rows),
                      "failed": [r["name"] for r in rows if r["status"] != 200]}))


if __name__ == "__main__":
    main()
