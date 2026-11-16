from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from graphlib import CycleError, TopologicalSorter
from typing import Any

import numpy as np


class Kind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"
    SERIES = "time_series"
    DISTRIBUTION = "distribution"
    OPERATOR = "operator"
    CONSTRAINT = "constraint"
    LATENT = "latent_state"
    KV = "kv_memory_reference"


class Flow(str, Enum):
    DATA = "data"
    PARAMETER = "parameter"
    CONSTRAINT = "constraint"
    GRADIENT = "gradient"
    TEMPORAL = "temporal"


class Relation(str, Enum):
    ASSIGN = "assignment"
    ADD = "additive_modulation"
    MULTIPLY = "multiplicative_modulation"
    FUNCTION = "functional_mapping"
    DISTRIBUTION = "distribution_modulation"
    CONSTRAINT = "constraint"
    FEEDBACK = "feedback"
    INTEGRATE = "temporal_integration"


@dataclass(frozen=True)
class Port:
    kind: Kind
    unit: str = "1"
    shape: tuple[int, ...] | None = None

    def validate(self, value: Any) -> None:
        if self.kind == Kind.SCALAR:
            if not np.isscalar(value) or not np.isfinite(value):
                raise TypeError("expected a finite scalar")
        elif self.kind in (Kind.VECTOR, Kind.TENSOR, Kind.SERIES, Kind.LATENT):
            a = np.asarray(value)
            if self.kind in (Kind.VECTOR, Kind.SERIES) and a.ndim != 1:
                raise TypeError("vector/time series requires one dimension")
            if not np.isfinite(a).all() or (self.shape is not None and a.shape != self.shape):
                raise TypeError(f"expected finite array with shape {self.shape}")
        else:
            if getattr(value, "value_kind", None) != self.kind:
                raise TypeError(f"expected typed {self.kind.value}")


@dataclass
class Node:
    name: str
    output: Port
    inputs: dict[str, Port] = field(default_factory=dict)
    compute: Callable[..., Any] | None = None


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    input: str
    flow: Flow = Flow.DATA
    relation: Relation = Relation.FUNCTION
    delay: int = 0


class Graph:
    """Deterministic graph, with one-tick explicit state for delayed feedback.

    Flow/relation describe operator semantics; node compute implements them. Gradient-flow
    edges are provenance only in this NumPy executor, not an autograd promise.
    """

    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self.nodes = {n.name: n for n in nodes}
        self.edges = edges
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate graph node")
        deps = {n: set() for n in self.nodes}
        attached = set()
        for e in edges:
            if e.source not in self.nodes or e.target not in self.nodes:
                raise ValueError("edge references unknown node")
            target = self.nodes[e.target]
            if e.input not in target.inputs:
                raise ValueError("edge references undeclared input")
            if self.nodes[e.source].output != target.inputs[e.input]:
                raise TypeError("edge kind/unit/shape mismatch")
            if (e.target, e.input) in attached:
                raise ValueError("multiple drivers for one input")
            attached.add((e.target, e.input))
            if e.delay not in (0, 1):
                raise ValueError("executor supports only zero or one tick delay")
            if e.delay and (e.flow != Flow.TEMPORAL or e.relation != Relation.FEEDBACK):
                raise ValueError("delay requires explicit temporal feedback")
            if not e.delay:
                deps[e.target].add(e.source)
        for n in nodes:
            if n.inputs and n.compute is None:
                raise ValueError("computed node requires function")
            if any((n.name, name) not in attached for name in n.inputs):
                raise ValueError("unconnected input")
        try:
            self.order = tuple(TopologicalSorter(deps).static_order())
        except CycleError as exc:
            raise ValueError("zero-delay cycle; add explicit delayed feedback") from exc

    def evaluate(self, sources: dict[str, Any], previous: dict[str, Any] | None = None):
        source_names = {n.name for n in self.nodes.values() if n.compute is None}
        if set(sources) != source_names:
            raise ValueError(f"sources must be exactly {sorted(source_names)}")
        result = {}
        for name in self.order:
            node = self.nodes[name]
            if node.compute is None:
                value = sources[name]
            else:
                inputs = {}
                for edge in self.edges:
                    if edge.target == name:
                        state = previous if edge.delay else result
                        if state is None or edge.source not in state:
                            raise ValueError("feedback requires explicit previous state")
                        inputs[edge.input] = state[edge.source]
                        node.inputs[edge.input].validate(inputs[edge.input])
                value = node.compute(**inputs)
            node.output.validate(value)
            result[name] = value
        return result

