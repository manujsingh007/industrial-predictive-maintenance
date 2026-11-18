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


def unit_summary(session: Session, unit_id: int) -> dict[str, int | None]:
    """Stored readings and predictions of a unit, for clients that resume (e.g. the replay)."""
    n_readings, last_cycle = count_readings(session, unit_id)
    n_predictions, last_prediction_cycle = session.execute(
        select(func.count(), func.max(PredictionRow.cycle)).where(PredictionRow.unit_id == unit_id)
    ).one()
    return {
        "unit_id": unit_id,
        "n_readings": n_readings,
        "last_cycle": last_cycle,
        "n_predictions": int(n_predictions),
        "last_prediction_cycle": last_prediction_cycle,
    }


def list_units(session: Session) -> list[dict]:
    """Every unit with stored readings: count, last cycle and its latest prediction (or None).

    One query: readings are aggregated per unit, and each unit's newest prediction is picked
    with a window function (row_number over created_at, then id), then left-joined.
    """
    readings = (
        select(
            SensorReadingRow.unit_id,
            func.count().label("n_readings"),
            func.max(SensorReadingRow.cycle).label("last_cycle"),
        )
        .group_by(SensorReadingRow.unit_id)
        .subquery()
    )
    ranked = select(
        PredictionRow,
        func.row_number()
        .over(
            partition_by=PredictionRow.unit_id,
            order_by=(PredictionRow.created_at.desc(), PredictionRow.id.desc()),
        )
        .label("rank"),
    ).subquery()
    latest = select(ranked).where(ranked.c.rank == 1).subquery()
    rows = session.execute(
        select(
            readings.c.unit_id,
            readings.c.n_readings,
            readings.c.last_cycle,
            latest.c.rul,
            latest.c.lower,
            latest.c.upper,
            latest.c.cycle,
            latest.c.created_at,
        )
        .outerjoin(latest, latest.c.unit_id == readings.c.unit_id)
        .order_by(readings.c.unit_id)
    ).all()
    return [
        {
            "unit_id": r.unit_id,
            "n_readings": r.n_readings,
            "last_cycle": r.last_cycle,
            "latest_prediction": None
            if r.rul is None
            else {
                "rul": r.rul,
                "lower": r.lower,
                "upper": r.upper,
                "cycle": r.cycle,
                "created_at": r.created_at,
            },
        }
        for r in rows
    ]


def get_readings(
    session: Session, unit_id: int, limit: int | None = None
) -> list[SensorReadingRow]:
    """A unit's stored readings in cycle order; with ``limit``, only the most recent ones."""
    query = select(SensorReadingRow).where(SensorReadingRow.unit_id == unit_id)
    if limit is None:
        return list(session.scalars(query.order_by(SensorReadingRow.cycle)).all())
    newest = session.scalars(query.order_by(SensorReadingRow.cycle.desc()).limit(limit)).all()
    return list(reversed(newest))


class Repository:
    """The API's view of the database. A FastAPI dependency, so unit tests can use a fake."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_readings(self, unit_id: int, readings: Sequence[dict]) -> int:
        return add_readings(self.session, unit_id, readings)

    def count_readings(self, unit_id: int) -> tuple[int, int | None]:
        return count_readings(self.session, unit_id)

    def get_history(self, unit_id: int) -> pd.DataFrame:
        return get_history(self.session, unit_id)

    def add_prediction(self, **values: object) -> PredictionRow:
        return add_prediction(self.session, **values)

    def list_predictions(self, unit_id: int, limit: int = 100) -> list[PredictionRow]:
        return list_predictions(self.session, unit_id, limit)

    def unit_known(self, unit_id: int) -> bool:
        return unit_known(self.session, unit_id)

    def unit_summary(self, unit_id: int) -> dict[str, int | None]:
        return unit_summary(self.session, unit_id)

    def list_units(self) -> list[dict]:
        return list_units(self.session)

    def get_readings(self, unit_id: int, limit: int | None = None) -> list[SensorReadingRow]:
        return get_readings(self.session, unit_id, limit)
