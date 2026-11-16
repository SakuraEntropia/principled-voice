import os
from pathlib import Path

import numpy as np
import pytest
import torch

from principled_voice.core.schema import Project
from principled_voice.synthesis.kokoro import KokoroBackend


@pytest.fixture(scope="module")
def backend():
    directory = os.environ.get("PV_MODEL_DIR")
    if not directory:
        pytest.skip("set PV_MODEL_DIR for actual frozen-model integration tests")
    torch.set_num_threads(4)
    return KokoroBackend(Path(directory), "cpu")


@pytest.mark.integration
def test_neutral_matches_upstream_and_repeat(backend):
    project = Project(text="The light is still here.")
    a = backend.synthesize(project)
    b = backend.synthesize(project)
    np.testing.assert_array_equal(a.audio, b.audio)
    ps = backend.phonemize(project)
    torch.manual_seed(project.seed)
    original = backend.model(ps, backend.voice[len(ps) - 1], return_output=True).audio.numpy()
    np.testing.assert_array_equal(a.audio, original)
    assert len(a.audio) == a.metadata["expected_samples"]
    assert len(a.audio) > 24000 and np.max(np.abs(a.audio)) > 0.02


@pytest.mark.integration
def test_native_duration_edit_changes_sample_count_without_losing_original(backend):
    project = Project(text="The light is still here.")
    a = backend.synthesize(project)
    durations = a.arrays["used_durations"].copy()
    durations[5] += 4
    edited = Project.model_validate({**project.model_dump(),
                                    "controls": {"token_durations": durations.tolist()}})
    b = backend.synthesize(edited)
    assert len(b.audio) - len(a.audio) == 4 * 600
    np.testing.assert_array_equal(a.arrays["generated_durations"], b.arrays["generated_durations"])
    assert b.arrays["used_durations"][5] != b.arrays["generated_durations"][5]


@pytest.mark.integration
def test_explicit_f0_is_decoder_input_and_duration_budget_is_enforced(backend):
    project = Project(text="The light is still here.",
                      controls={"f0_hz": {"keys": [{"time": 0, "value": 200}]}})
    a = backend.synthesize(project)
    voiced = a.arrays["generated_f0"] > 10
    np.testing.assert_array_equal(a.arrays["used_f0"][voiced], np.full(voiced.sum(), 200))
    token_count = len(a.arrays["used_durations"])
    huge = Project.model_validate({**project.model_dump(),
                                   "controls": {"token_durations": [100] * token_count}})
    with pytest.raises(ValueError, match="budget"):
        backend.synthesize(huge)
