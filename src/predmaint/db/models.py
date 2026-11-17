"""SQLAlchemy 2 models. The schema itself is created by Alembic migrations, not by these classes.

- ``sensor_readings``: one row per engine cycle; (unit_id, cycle) is unique, so the same
  reading can never be stored twice. The unique constraint's index also serves the
  "history of one unit, ordered by cycle" query.
- ``predictions``: an append-only log of every prediction made from stored history.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Double,
    Identity,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from predmaint.data.load import SENSOR_COLUMNS, SETTING_COLUMNS

READING_VALUE_COLUMNS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]


class Base(DeclarativeBase):
    pass


class SensorReadingRow(Base):
    __tablename__ = "sensor_readings"
    __table_args__ = (UniqueConstraint("unit_id", "cycle", name="uq_sensor_readings_unit_cycle"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    unit_id: Mapped[int] = mapped_column(Integer)
    cycle: Mapped[int] = mapped_column(Integer)
    setting_1: Mapped[float] = mapped_column(Double)
    setting_2: Mapped[float] = mapped_column(Double)
    setting_3: Mapped[float] = mapped_column(Double)
    s_1: Mapped[float] = mapped_column(Double)
    s_2: Mapped[float] = mapped_column(Double)
    s_3: Mapped[float] = mapped_column(Double)
    s_4: Mapped[float] = mapped_column(Double)
    s_5: Mapped[float] = mapped_column(Double)
    s_6: Mapped[float] = mapped_column(Double)
    s_7: Mapped[float] = mapped_column(Double)
    s_8: Mapped[float] = mapped_column(Double)
    s_9: Mapped[float] = mapped_column(Double)
    s_10: Mapped[float] = mapped_column(Double)
    s_11: Mapped[float] = mapped_column(Double)
    s_12: Mapped[float] = mapped_column(Double)
    s_13: Mapped[float] = mapped_column(Double)
    s_14: Mapped[float] = mapped_column(Double)
    s_15: Mapped[float] = mapped_column(Double)
    s_16: Mapped[float] = mapped_column(Double)
    s_17: Mapped[float] = mapped_column(Double)
    s_18: Mapped[float] = mapped_column(Double)
    s_19: Mapped[float] = mapped_column(Double)
    s_20: Mapped[float] = mapped_column(Double)
    s_21: Mapped[float] = mapped_column(Double)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PredictionRow(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_unit_created", "unit_id", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    unit_id: Mapped[int] = mapped_column(Integer)
    cycle: Mapped[int] = mapped_column(Integer)  # last stored cycle the prediction is based on
    rul: Mapped[float] = mapped_column(Double)
    lower: Mapped[float] = mapped_column(Double)
    upper: Mapped[float] = mapped_column(Double)
    model_name: Mapped[str] = mapped_column(String(200))
    model_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
