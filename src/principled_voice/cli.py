from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="pv")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("project", type=Path)
    synth = sub.add_parser("render")
    synth.add_argument("project", type=Path)
    synth.add_argument("--model-dir", type=Path, default=Path("models/kokoro"))
    synth.add_argument("--out", type=Path, required=True)
    synth.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    inverse = sub.add_parser("inverse")
    inverse.add_argument("--pitch-st", type=float, required=True)
    inverse.add_argument("--n", type=float, required=True)
    inverse.add_argument("--inhibition", type=float, default=0)
    sub.add_parser("doctor")
    kv = sub.add_parser("kv-experiment")
    kv.add_argument("--model-dir", type=Path, default=Path("models/parler"))
    kv.add_argument("--out", type=Path, required=True)
    kv.add_argument("--max-tokens", type=int, default=350)
    kv.add_argument("--sampling", action="store_true", help="Use the checkpoint's native sampling mode")
    args = parser.parse_args()
    if args.command == "doctor":
        import platform

        import torch

        print(json.dumps({"python": platform.python_version(), "machine": platform.machine(),
                          "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
                          "cuda_available": torch.cuda.is_available()}, indent=2))
    elif args.command == "inverse":
        from .core.coupling import inverse_proxy

        print(json.dumps(inverse_proxy(args.pitch_st, args.n, args.inhibition), indent=2))
    elif args.command == "kv-experiment":
        from .research.parler import run_experiment

        print(json.dumps(run_experiment(args.model_dir, args.out, args.max_tokens, args.sampling), indent=2))
    else:
        from .synthesis.kokoro import KokoroBackend, load_project

        project = load_project(args.project)
        if args.command == "validate":
            print(project.model_dump_json(indent=2))
        else:
            import torch

            from .pipeline import render

            torch.set_num_threads(4)
            backend = KokoroBackend(args.model_dir, args.device)
            manifest = render(project, backend, args.out)
            print(json.dumps({"status": manifest["status"], "output": str(args.out.resolve()),
                              "rtf": manifest["real_time_factor"]}, indent=2))
            if manifest["status"] == "constraint_violation":
                raise SystemExit(2)


if __name__ == "__main__":
    main()
