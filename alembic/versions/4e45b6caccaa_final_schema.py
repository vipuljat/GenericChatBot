"""final schema

Revision ID: 4e45b6caccaa
Revises: d08f68d81bec
Create Date: 2025-12-16 13:43:22.278448
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "4e45b6caccaa"
down_revision: Union[str, Sequence[str], None] = "d08f68d81bec"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop FK first (type change requires it)
    op.drop_constraint(
        op.f("answers_attempter_by_id_fkey"),
        "answers",
        type_="foreignkey",
    )

    # Convert UUID -> String safely
    op.alter_column(
        "answers",
        "attempter_by_id",
        existing_type=postgresql.UUID(as_uuid=True),
        type_=sa.String(),
        existing_nullable=True,
        postgresql_using="attempter_by_id::text",
    )

    # Enforce NOT NULL on answer_data
    op.alter_column(
        "answers",
        "answer_data",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
    )

    # Recreate FK to employees.employee_id (String -> String)
    op.create_foreign_key(
        "answers_attempter_by_id_fkey",
        "answers",
        "employees",
        ["attempter_by_id"],
        ["employee_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # Drop FK first
    op.drop_constraint(
        "answers_attempter_by_id_fkey",
        "answers",
        type_="foreignkey",
    )

    # Convert String -> UUID safely
    op.alter_column(
        "answers",
        "attempter_by_id",
        existing_type=sa.String(),
        type_=postgresql.UUID(as_uuid=True),
        existing_nullable=True,
        postgresql_using="attempter_by_id::uuid",
    )

    # Revert NOT NULL
    op.alter_column(
        "answers",
        "answer_data",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=True,
    )

    # Restore FK to employees.id (UUID -> UUID)
    op.create_foreign_key(
        "answers_attempter_by_id_fkey",
        "answers",
        "employees",
        ["attempter_by_id"],
        ["id"],
        ondelete="SET NULL",
    )
