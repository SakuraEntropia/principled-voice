from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Key(StrictModel):
    time: float = Field(ge=0, le=1)
    value: float


class Curve(StrictModel):
    """Normalized time; linear or smoothstep interpolation, no implied Bezier handles."""

    keys: list[Key] = Field(min_length=1)
    interpolation: Literal["linear", "smooth"] = "linear"

    @model_validator(mode="after")
    def ordered(self):
        if any(a.time >= b.time for a, b in zip(self.keys, self.keys[1:])):
            raise ValueError("curve times must be strictly increasing")
        return self

    def sample(self, count: int) -> np.ndarray:
        if count < 1:
            raise ValueError("sample count must be positive")
        times = np.array([k.time for k in self.keys])
        values = np.array([k.value for k in self.keys])
        t = np.linspace(0, 1, count)
        if len(times) == 1:
            return np.full(count, values[0])
        i = np.searchsorted(times, t, side="right").clip(1, len(times) - 1)
        u = ((t - times[i - 1]) / (times[i] - times[i - 1])).clip(0, 1)
        if self.interpolation == "smooth":
            u = u * u * (3 - 2 * u)
        return values[i - 1] * (1 - u) + values[i] * u


class Performance(StrictModel):
    # Deliberately neutral engineering defaults, not Kororu character attributes.
    arousal: float = Field(default=0, ge=-1, le=1)
    inhibition: float = Field(default=0, ge=0, le=1)


class Controls(StrictModel):
    speed: float = Field(default=1, ge=0.5, le=2)
    pitch_semitones: float = Field(default=0, ge=-12, le=12)
    pitch_curve: Curve | None = None
    f0_hz: Curve | None = None
    # N is a learned acoustic condition. It is NOT calibrated energy or pressure.
    n_offset: float = Field(default=0, ge=-2, le=2)
    n_curve: Curve | None = None
    output_gain_db: float = Field(default=0, ge=-24, le=12)
    token_durations: list[Annotated[int, Field(ge=1, le=100)]] | None = None
    performance: Performance = Field(default_factory=Performance)

    @model_validator(mode="after")
    def valid_f0(self):
        if self.f0_hz and any(not 40 <= k.value <= 800 for k in self.f0_hz.keys):
            raise ValueError("F0 conditioning must be 40–800 Hz")
        if self.pitch_curve and any(abs(k.value) > 12 for k in self.pitch_curve.keys):
            raise ValueError("pitch curve exceeds ±12 semitones")
        if self.n_curve and any(abs(k.value) > 2 for k in self.n_curve.keys):
            raise ValueError("N curve exceeds ±2 experimental units")
        if self.f0_hz and (self.pitch_semitones != 0 or self.pitch_curve is not None):
            raise ValueError("absolute F0 and relative pitch edits conflict; choose one")
        if self.token_durations is not None and self.speed != 1:
            raise ValueError("explicit duration and speed conflict; choose one")
        return self


class AcousticLock(StrictModel):
    parameter: Literal["f0_hz", "rms_dbfs"]
    target: float
    tolerance: float = Field(gt=0)
    strength: Literal["hard", "soft"] = "hard"


class Project(StrictModel):
    schema_version: Literal[1] = 1
    character: Literal["kororu"] = "kororu"
    identity_status: Literal["temporary_voice"] = "temporary_voice"
    scene: str = "technical_test"
    shot: str = "test_001"
    take: str = "take_001"
    fps: float = Field(default=24, gt=0, le=120)
    start_frame: int = Field(default=0, ge=0)
    text: str = Field(min_length=1, max_length=1000)
    language: Literal["en-us"] = "en-us"
    phonemes: str | None = Field(default=None, min_length=1, max_length=500)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    voice: Literal["af_heart"] = "af_heart"
    controls: Controls = Field(default_factory=Controls)
    locks: list[AcousticLock] = Field(default_factory=list)

    @model_validator(mode="after")
    def lock_conflicts(self):
        names = [lock.parameter for lock in self.locks]
        if len(names) != len(set(names)):
            raise ValueError("duplicate acoustic locks")
        if "rms_dbfs" in names:
            raise ValueError("waveform RMS lock is unsupported; use inverse diagnostics first")
        if "f0_hz" in names and (
            self.controls.f0_hz or self.controls.pitch_curve or self.controls.pitch_semitones
        ):
            raise ValueError("F0 lock conflicts with explicit pitch edits")
        for lock in self.locks:
            if lock.parameter == "f0_hz" and not 40 <= lock.target <= 800:
                raise ValueError("F0 lock outside 40–800 Hz domain")
        return self

