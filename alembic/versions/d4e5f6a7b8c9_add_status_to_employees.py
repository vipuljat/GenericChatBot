"""add status to employees (active / draft / deleted)

Revision ID: d4e5f6a7b8c9
Revises: a9b8c7d6e5f4
Create Date: 2026-11-12
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'a9b8c7d6e5f4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'employees',
        sa.Column('status', sa.String(), nullable=False, server_default='active'),
    )


def downgrade() -> None:
    op.drop_column('employees', 'status')
