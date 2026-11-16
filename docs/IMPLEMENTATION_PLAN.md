# M0/M1 implementation plan

Goal: real local voice pipeline plus falsifiable control and cache experiments.
Architecture: typed graph and constrained proxy solver independent of frozen backends.
Stack: Python 3.12, PyTorch CPU/MPS probe, Kokoro, Parler, NumPy/SciPy, JSON, WAV.
Spec: DESIGN.md and DESIGN.md. Current user explicitly requested execution.

- [x] Audit primary papers, official repositories, licenses and runtime; save sources and hashes.
- [x] Implement core/schema.py (validated curves/project), graph.py (typed delayed DAG),
      coupling.py (operator/distribution modulation and inverse feasibility).
- [x] Test invalid controls, duplicate times, graph mismatch/cycle/feedback, parameter-of-parameter
      response, inverse feasible/infeasible cases and non-destructive overrides.
- [x] Implement synthesis/kokoro.py native duration/F0/N controls; keep generated and edited states.
- [x] Implement analysis.py independent waveform measurement and pipeline.py versioned manifests.
- [x] Run real baseline, F0 curve, duration, N, inhibition and constrained controls on this machine.
- [x] Implement research/kv.py metadata validation, prefix/window mask, separate K/V/joint/rank patches.
- [x] Run numerical no-op and incompatibility tests, then actual frozen Parler cache/audio comparisons.
- [x] Record actual results, failures, compatibility, blockers and exact reproducible commands.
- [x] Package install/run documentation and a local git snapshot; do not publish anything.

