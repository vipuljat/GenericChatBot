"""added ChatbotAccess Table

Revision ID: 2c8b0dae2ab5
Revises: c8b6c40b61a7
Create Date: 2025-12-24 01:40:52.550117
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2c8b0dae2ab5'
down_revision: Union[str, Sequence[str], None] = 'c8b6c40b61a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # 1️⃣ Create chatbot_access table
    op.create_table(
        'chatbot_access',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('chatbot_id', sa.UUID(), nullable=False),
        sa.Column('employee_id', sa.String(), nullable=True),
        sa.Column('allowed_users', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['chatbot_id'], ['chatbots.chatbot_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.employee_id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # 2️⃣ Drop FK BEFORE type change
    op.drop_constraint(
        op.f('chatbot_permissions_employee_id_fkey'),
        'chatbot_permissions',
        type_='foreignkey'
    )

    # 3️⃣ Convert employee_id from UUID → JSONB safely
    op.alter_column(
        'chatbot_permissions',
        'employee_id',
        existing_type=sa.UUID(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        postgresql_using="to_jsonb(employee_id)",
        existing_nullable=False
    )


def downgrade() -> None:
    """Downgrade schema."""

    # 1️⃣ Convert JSONB → UUID (assumes single-value JSON)
    op.alter_column(
        'chatbot_permissions',
        'employee_id',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.UUID(),
        postgresql_using="employee_id::text::uuid",
        existing_nullable=False
    )

    # 2️⃣ Restore FK
    op.create_foreign_key(
        op.f('chatbot_permissions_employee_id_fkey'),
        'chatbot_permissions',
        'employees',
        ['employee_id'],
        ['employee_id'],
        ondelete='CASCADE'
    )

    # 3️⃣ Drop chatbot_access table
    op.drop_table('chatbot_access')
