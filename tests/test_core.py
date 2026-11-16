import numpy as np
import pytest
from pydantic import ValidationError

from principled_voice.analysis import analyze
from principled_voice.core.coupling import EditedParameter, evaluate_performance, inverse_proxy
from principled_voice.core.graph import Edge, Flow, Graph, Kind, Node, Port, Relation
from principled_voice.core.schema import Curve, Project


def test_curve_order_domain_and_interpolation():
    c = Curve(keys=[{"time": 0, "value": 0}, {"time": 1, "value": 2}])
    np.testing.assert_allclose(c.sample(3), [0, 1, 2])
    with pytest.raises(ValidationError):
        Curve(keys=[{"time": 0.5, "value": 0}, {"time": 0.5, "value": 2}])
    with pytest.raises(ValidationError):
        Project(text="test", controls={"f0_hz": {"keys": [{"time": 0, "value": 900}]}})
    with pytest.raises(ValidationError):
        Project(text="test", controls={"pitch_semitones": 2, "f0_hz": c.model_dump()})


def test_unknown_project_fields_and_conflicting_lock_fail():
    with pytest.raises(ValidationError):
        Project(text="test", invented_character_age=12)
    with pytest.raises(ValidationError):
        Project(text="test", controls={"pitch_semitones": 2},
                locks=[{"parameter": "f0_hz", "target": 200, "tolerance": 5}])
    with pytest.raises(ValidationError):
        Project(text="test", locks=[{"parameter": "rms_dbfs", "target": -20, "tolerance": 1}])


def test_graph_rejects_unit_mismatch_and_cycles():
    s = Port(Kind.SCALAR)
    with pytest.raises(TypeError):
        Graph([Node("a", Port(Kind.SCALAR, "Hz")),
               Node("b", s, {"x": Port(Kind.SCALAR, "dB")}, lambda x: x)], [Edge("a", "b", "x")])
    with pytest.raises(ValueError, match="cycle"):
        Graph([Node("a", s, {"x": s}, lambda x: x), Node("b", s, {"x": s}, lambda x: x)],
              [Edge("a", "b", "x"), Edge("b", "a", "x")])


def test_vector_and_series_have_real_shape_checks():
    with pytest.raises(TypeError):
        Port(Kind.VECTOR).validate(2)
    with pytest.raises(TypeError):
        Port(Kind.SERIES, shape=(3,)).validate([1, 2])
    with pytest.raises(TypeError):
        Port(Kind.SCALAR).validate(float("nan"))
    Port(Kind.VECTOR, shape=(2,)).validate([1, 2])


def test_explicit_feedback_reads_previous_not_current():
    s = Port(Kind.SCALAR)
    graph = Graph([Node("state", s, {"last": s}, lambda last: last + 1)],
                  [Edge("state", "state", "last", Flow.TEMPORAL, Relation.FEEDBACK, delay=1)])
    assert graph.evaluate({}, {"state": 4}) == {"state": 5}
    with pytest.raises(ValueError, match="previous"):
        graph.evaluate({})


def test_operator_and_distribution_modulation():
    open_ = evaluate_performance(0.8, 0)
    inhibited = evaluate_performance(0.8, 1)
    assert open_["emotion_to_physiology"]["gain"] == 1
    assert inhibited["emotion_to_physiology"]["gain"] == 0
    assert open_["prosody_distribution"]["pitch_mean_st"] > 1
    assert open_["prosody_distribution"]["pitch_std_st"] > inhibited["prosody_distribution"]["pitch_std_st"]
    assert inhibited["tension_proxy"] == 0


def test_inverse_keeps_targets_and_reports_impossibility():
    feasible = inverse_proxy(1, 0.1)
    assert feasible["feasible"]
    np.testing.assert_allclose(feasible["achieved"], [1, 0.1], atol=0.01)
    impossible = inverse_proxy(10, 5)
    assert not impossible["feasible"]
    assert impossible["target"] == [10, 5]
    assert impossible["suggested_relaxation"]


def test_override_does_not_destroy_generated_state():
    p = EditedParameter(200, 220, True)
    assert p.value == 220 and p.generated == 200 and p.locked


def test_independent_pitch_estimator_on_known_signal():
    sr = 24000
    x = 0.2 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    m = analyze(x, sr)
    assert abs(m["f0_median_hz"] - 200) < 2
    assert abs(m["rms_dbfs"] - 20 * np.log10(0.2 / np.sqrt(2))) < 0.1

