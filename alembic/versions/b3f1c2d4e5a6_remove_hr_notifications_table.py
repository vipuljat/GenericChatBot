"""remove hr_notifications table

Revision ID: b3f1c2d4e5a6
Revises: e7d30ca26a1d
Create Date: 2026-02-20
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'b3f1c2d4e5a6'
down_revision: Union[str, Sequence[str], None] = 'e7d30ca26a1d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('DROP TABLE IF EXISTS hr_notifications')


def downgrade() -> None:
    op.create_table(
        'hr_notifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('employee_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('chatbot_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chatbot_type', sa.String(), nullable=False),
        sa.Column('priority', sa.String(), nullable=False, server_default='high'),
        sa.Column('status', sa.String(), nullable=False, server_default='unread'),
        sa.Column('issue_summary', sa.Text(), nullable=False),
        sa.Column('conversation_context', postgresql.JSONB(), nullable=True),
        sa.Column('employee_feedback', sa.Text(), nullable=True),
        sa.Column('assigned_to', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['chatbot_id'], ['chatbots.chatbot_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['assigned_to'], ['employees.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
