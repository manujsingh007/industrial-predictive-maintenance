"""Database operations used by the API. Each function works inside the caller's session."""

from collections.abc import Sequence

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from predmaint.db.models import READING_VALUE_COLUMNS, PredictionRow, SensorReadingRow


class DuplicateCyclesError(Exception):
    """Some submitted cycles are already stored for this unit."""

    def __init__(self, unit_id: int, cycles: Sequence[int]) -> None:
        self.unit_id, self.cycles = unit_id, sorted(cycles)
        super().__init__(f"unit {unit_id} already has readings for cycles {self.cycles[:10]}")


def add_readings(session: Session, unit_id: int, readings: Sequence[dict]) -> int:
    """Insert readings for one unit, all or nothing.

    Raises:
        DuplicateCyclesError: If any cycle is already stored. Nothing is inserted. The unique
            constraint also catches a concurrent insert between the check and the commit.
    """
    cycles = [int(r["cycle"]) for r in readings]
    existing = session.scalars(
        select(SensorReadingRow.cycle).where(
            SensorReadingRow.unit_id == unit_id, SensorReadingRow.cycle.in_(cycles)
        )
    ).all()
    if existing:
        raise DuplicateCyclesError(unit_id, existing)
    session.add_all(SensorReadingRow(unit_id=unit_id, **r) for r in readings)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise DuplicateCyclesError(unit_id, cycles) from exc
    return len(readings)


def count_readings(session: Session, unit_id: int) -> tuple[int, int | None]:
    """(number of stored readings, last cycle) for a unit."""
    count, last = session.execute(
        select(func.count(), func.max(SensorReadingRow.cycle)).where(
            SensorReadingRow.unit_id == unit_id
        )
    ).one()
    return int(count), last


def get_history(session: Session, unit_id: int) -> pd.DataFrame:
    """All stored readings of a unit as a DataFrame (unit, cycle, settings, sensors), by cycle.

    Returns an empty DataFrame when the unit is unknown.
    """
    rows = session.scalars(
        select(SensorReadingRow)
        .where(SensorReadingRow.unit_id == unit_id)
        .order_by(SensorReadingRow.cycle)
    ).all()
    columns = ["unit", "cycle", *READING_VALUE_COLUMNS]
    records = [[r.unit_id, r.cycle, *(getattr(r, c) for c in READING_VALUE_COLUMNS)] for r in rows]
    return pd.DataFrame(records, columns=columns)


def add_prediction(session: Session, **values: object) -> PredictionRow:
    """Store a prediction and return it with its id and created_at filled in."""
    row = PredictionRow(**values)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_predictions(session: Session, unit_id: int, limit: int = 100) -> list[PredictionRow]:
    """A unit's stored predictions, newest first (ties broken by id)."""
    return list(
        session.scalars(
            select(PredictionRow)
            .where(PredictionRow.unit_id == unit_id)
            .order_by(PredictionRow.created_at.desc(), PredictionRow.id.desc())
            .limit(limit)
        ).all()
    )


def unit_known(session: Session, unit_id: int) -> bool:
    """True if the unit has any stored readings or predictions."""
    for model in (SensorReadingRow, PredictionRow):
        if session.scalar(select(model.id).where(model.unit_id == unit_id).limit(1)) is not None:
            return True
    return False
