"""Upgrade chat_history column from TEXT to JSONB

Revision ID: c8b6c40b61a7
Revises: a193ef60a619
Create Date: 2025-12-22 07:21:59.571982
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c8b6c40b61a7"
down_revision: Union[str, Sequence[str], None] = "a193ef60a619"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Convert chat_history from TEXT -> JSONB.
    Existing values must contain valid JSON.
    """
    # Use raw SQL because PostgreSQL requires USING for JSONB casts
    op.execute(
        """
        ALTER TABLE answers
        ALTER COLUMN chat_history
        TYPE JSONB
        USING chat_history::jsonb
        """
    )


def downgrade() -> None:
    """
    Convert chat_history from JSONB -> TEXT.
    """
    op.execute(
        """
        ALTER TABLE answers
        ALTER COLUMN chat_history
        TYPE TEXT
        USING chat_history::text
        """
    )
