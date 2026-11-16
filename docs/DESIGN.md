# Principled Voice M0/M1 design · 2026-11-16

Scope approved by the current user request: implement the referenced specification's M0,
real-waveform M1, and initial explicit-control / KV experiments locally. No publishing,
modification of the animation archive, training on private recordings, or character invention.
This design narrows the implementation sequence, not the long-term requirements.

## Decisions

1. Keep a frozen Kokoro-82M v1.0 as the low-memory temporary English voice backend.
   Native F0/N conditioning and token alignment are inspectable. It is not autoregressive.
2. Use a separately loaded Parler-TTS tiny checkpoint for real autoregressive cache tests.
   No cache translation between models. Do not assume linear emotion directions.
3. Use a src-layout Python package, CLI first. M5 Qt DCC editor is deferred; a CLI is not a GUI.
4. Store declarative projects with character/scene/shot/take, seed, curves, locks, overrides,
   and reproducibility manifests. Do not bake edits into original generated parameters.
5. Kororu identity has no supplied demographic, personality, language or timbre values.
   All generated voice examples are Temporary Voice and technical test text.

## Typed parameter graph

Ports carry value-kind, units and shape. Edges separately declare data, parameter,
constraint, gradient and temporal flow; relation labels include assignment, additive,
multiplicative, functional, distribution, feedback and integration. Evaluation rejects
undeclared inputs, type/unit/shape mismatch and zero-delay cycles. Feedback reads an
explicit previous evaluation snapshot; no hidden algebraic loops.

Concrete coupling: inhibition changes an emotion→physiology operator's gain; arousal
passes through that operator into bounded dimensionless tension/pressure proxies;
those proxies change a prosody distribution's mean AND variance. Acoustic controls are
derived from that distribution. These functions are engineering hypotheses, not measured
human physiology or learned emotion models. User overrides retain the generated values.

## Constraints and inverse path

Use bounded least squares to recover upstream proxy state from F0/energy targets. Return
residual, bounds and suggested relaxation on infeasibility. Never silently remove locks.
Native F0 locks mean requested decoder conditioning; independently measured waveform F0
may deviate. Such deviation is reported, not labeled successful constraint satisfaction.
Hard output loudness targets and unsupported physical locks are not implied by N control.

## Synthesis and measurement

Project → validated controls → evaluated graph → native duration alignment → F0/N edits →
frozen neural decoder → WAV + token alignment + raw conditioning NPZ + manifest + analysis.
No output pitch shifting or waveform time stretching. Optional output gain is explicitly a
post effect. Curve times use normalized utterance time in [0,1]; token durations are decoder
tokens (including stress/space/punctuation), not forced-aligned ground-truth phonemes.
The Kokoro duration frame is 600 samples at 24kHz; measure and test this assumption.

F0 is estimated independently from waveform autocorrelation/pYIN; RMS, duration, spectral
centroid, elapsed time and process RSS are measured. ASR, identity and emotion scores remain
unmeasured unless the relevant evaluators and valid references are actually run.

## KV safety and experimental boundary

Cache metadata includes model revision, layer, tensor layout, dtype/device, exact position
IDs, prompt length and head layout. Reject incompatible swaps. Copy caches before editing;
preserve original query positions and full cache length. Prefix/window masking is distinct
from cache replacement. Compare no-op, K-only, V-only, joint, low-rank, head/layer patches and
activation baseline on matched text/seed. Native descriptions are a baseline only. Cache
shape tests establish engineering validity, not an emotional effect. Audio evaluation needs
matched references, held-out texts and calibrated perceptual/identity checks before M3 passes.

## Acceptance and limits

M0: auditable primary sources, separate code/weight/data licenses, decisions and blockers.
M1: real neural speech on this Mac, local rerun command, no fabricated Kororu voice identity,
reproducibility and evidence. M2–M4 receive tested foundations but are not claimed complete.
M5 editor and M6 production validation remain future milestones.

