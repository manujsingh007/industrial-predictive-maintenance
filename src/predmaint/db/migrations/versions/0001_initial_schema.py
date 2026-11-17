"""Initial schema: sensor_readings and predictions.

Revision ID: 0001
Revises:
Create Date: 2026-11-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VALUE_COLUMNS = [f"setting_{i}" for i in range(1, 4)] + [f"s_{i}" for i in range(1, 22)]


def upgrade() -> None:
    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        *[sa.Column(name, sa.Double(), nullable=False) for name in VALUE_COLUMNS],
        sa.Column(
            "ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("unit_id", "cycle", name="uq_sensor_readings_unit_cycle"),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        sa.Column("rul", sa.Double(), nullable=False),
        sa.Column("lower", sa.Double(), nullable=False),
        sa.Column("upper", sa.Double(), nullable=False),
        sa.Column("model_name", sa.String(200), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_predictions_unit_created", "predictions", ["unit_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_predictions_unit_created", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("sensor_readings")
