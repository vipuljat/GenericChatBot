"""add updated_at to answers table

Revision ID: a9b8c7d6e5f4
Revises: c1d2e3f4a5b6
Create Date: 2026-04-18
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'a9b8c7d6e5f4'
down_revision: Union[str, Sequence[str], None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'answers' AND column_name = 'updated_at'"
    ))
    if not result.fetchone():
        op.add_column(
            'answers',
            sa.Column(
                'updated_at',
                sa.DateTime(timezone=True),
                server_default=sa.text('now()'),
                nullable=True,
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'answers' AND column_name = 'updated_at'"
    ))
    if result.fetchone():
        op.drop_column('answers', 'updated_at')
