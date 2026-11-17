"""Request and response schemas for the prediction API (Pydantic v2).

A request carries one engine's raw readings in the C-MAPSS layout: cycle, three operational
settings and 21 sensors. The model uses only the 14 kept sensors (ADR 0001), but the schema
accepts the full raw record so clients can send readings exactly as the engine produces them.
"""

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_READINGS = 1000
"""Most readings accepted per request.

FD001's longest engine has 362 cycles, so 1000 leaves ample headroom while bounding request size
and CPU time per call. Only the last 30 cycles affect the window features, but the full history
is accepted, so ``n_cycles_seen`` is reported correctly.
"""


class SensorReading(BaseModel):
    """One cycle of raw readings from one engine."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cycle: int = Field(ge=1, description="Cycle number, starting at 1")
    setting_1: float
    setting_2: float
    setting_3: float
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float


class PredictRequest(BaseModel):
    """One engine's readings, oldest first; RUL is predicted at the last reading."""

    model_config = ConfigDict(extra="forbid")

    unit_id: int = Field(ge=1, description="Engine identifier")
    readings: list[SensorReading] = Field(
        min_length=1,
        max_length=MAX_READINGS,
        description=f"1 to {MAX_READINGS} readings, sorted by strictly increasing cycle",
    )

    @model_validator(mode="after")
    def cycles_strictly_increasing(self) -> "PredictRequest":
        cycles = [r.cycle for r in self.readings]
        duplicates = sorted(c for c, n in Counter(cycles).items() if n > 1)
        if duplicates:
            raise ValueError(f"duplicate cycle values: {duplicates[:5]}")
        if any(b <= a for a, b in zip(cycles, cycles[1:], strict=False)):
            raise ValueError("readings must be sorted by strictly increasing cycle")
        return self


class PredictResponse(BaseModel):
    """Predicted RUL at the last reading, with its 90% prediction interval."""

    unit_id: int
    rul: float = Field(description="Predicted remaining useful life in cycles (>= 0)")
    lower: float = Field(description="Lower bound of the 90% interval (>= 0)")
    upper: float = Field(description="Upper bound of the 90% interval")
    n_cycles_seen: int = Field(description="Number of readings the prediction is based on")
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Service status; ``model_name``/``model_version`` are null when no model is loaded."""

    status: str
    model_name: str | None = None
    model_version: str | None = None
