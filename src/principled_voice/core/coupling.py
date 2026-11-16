from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import least_squares

from .graph import Edge, Flow, Graph, Kind, Node, Port, Relation


@dataclass(frozen=True)
class EmotionPhysiologyOperator:
    value_kind: ClassVar[Kind] = Kind.OPERATOR
    gain: float

    def __call__(self, arousal: float) -> float:
        return float(np.tanh(self.gain * arousal))


@dataclass(frozen=True)
class ProsodyDistribution:
    value_kind: ClassVar[Kind] = Kind.DISTRIBUTION
    pitch_mean_st: float
    pitch_std_st: float
    n_mean: float


def distribution(tension: float, pressure: float) -> ProsodyDistribution:
    return ProsodyDistribution(3 * tension, 0.2 + 0.5 * abs(tension), 0.4 * pressure)


def performance_graph() -> Graph:
    s = Port(Kind.SCALAR)
    op = Port(Kind.OPERATOR)
    dist = Port(Kind.DISTRIBUTION)
    return Graph([
        Node("arousal", s), Node("inhibition", s),
        Node("emotion_to_physiology", op, {"inhibition": s},
             lambda inhibition: EmotionPhysiologyOperator(1 - inhibition)),
        Node("tension_proxy", s, {"operator": op, "arousal": s},
             lambda operator, arousal: operator(arousal)),
        Node("pressure_proxy", s, {"tension": s}, lambda tension: 0.7 * tension),
        Node("prosody_distribution", dist, {"tension": s, "pressure": s}, distribution),
    ], [
        Edge("inhibition", "emotion_to_physiology", "inhibition", Flow.PARAMETER,
             Relation.MULTIPLY),
        Edge("emotion_to_physiology", "tension_proxy", "operator", Flow.PARAMETER),
        Edge("arousal", "tension_proxy", "arousal"),
        Edge("tension_proxy", "pressure_proxy", "tension"),
        Edge("tension_proxy", "prosody_distribution", "tension", Flow.PARAMETER,
             Relation.DISTRIBUTION),
        Edge("pressure_proxy", "prosody_distribution", "pressure", Flow.PARAMETER,
             Relation.DISTRIBUTION),
    ])


def evaluate_performance(arousal: float, inhibition: float) -> dict:
    if not -1 <= arousal <= 1 or not 0 <= inhibition <= 1:
        raise ValueError("performance outside domain")
    values = performance_graph().evaluate({"arousal": arousal, "inhibition": inhibition})
    return {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v for k, v in values.items()}


def inverse_proxy(target_pitch_st: float, target_n: float, inhibition: float = 0) -> dict:
    """Recover arousal and bounded residual phonation from acoustic proxy targets.

    This does NOT infer biological variables or prove waveform constraints. Residual
    phonation is the remaining freedom when pitch is locked. Scaling defines tolerances.
    """
    if not np.isfinite([target_pitch_st, target_n, inhibition]).all() or not 0 <= inhibition <= 1:
        raise ValueError("invalid inverse target")

    def forward(z):
        t = np.tanh((1 - inhibition) * z[0])
        return np.array([3 * t + z[1], 0.28 * t + 0.4 * z[2]])

    target = np.array([target_pitch_st, target_n])
    tolerances = np.array([0.01, 0.001])
    result = least_squares(lambda z: (forward(z) - target) / tolerances, [0, 0, 0],
                           bounds=([-1, -1, -1], [1, 1, 1]), max_nfev=1000,
                           ftol=1e-12, gtol=1e-12, xtol=1e-12)
    achieved = forward(result.x)
    residual = achieved - target
    feasible = bool(np.all(np.abs(residual) <= tolerances))
    return {"domain": "untrained_acoustic_proxy", "feasible": feasible,
            "state": dict(zip(["arousal", "tension_residual", "pressure_residual"], result.x.tolist())),
            "target": target.tolist(), "achieved": achieved.tolist(),
            "residual": residual.tolist(), "solver_success": bool(result.success),
            "suggested_relaxation": None if feasible else
                "Reduce pitch/N target or explicitly enlarge the proxy residual bounds."}


@dataclass(frozen=True)
class EditedParameter:
    generated: float
    override: float | None = None
    locked: bool = False

    @property
    def value(self) -> float:
        return self.generated if self.override is None else self.override
