"""foreign key update in people analyzer table

Revision ID: a193ef60a619
Revises: f984cfafcdd8
Create Date: 2025-12-21 14:39:02.107932
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a193ef60a619'
down_revision = 'f984cfafcdd8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop existing FK constraints
    op.drop_constraint('people_analyzer_employee_id_fkey', 'people_analyzer', type_='foreignkey')
    op.drop_constraint('people_analyzer_created_by_fkey', 'people_analyzer', type_='foreignkey')

    # Alter columns from UUID to String
    op.alter_column('people_analyzer', 'employee_id',
               existing_type=sa.UUID(),
               type_=sa.String(),
               existing_nullable=True)
    op.alter_column('people_analyzer', 'created_by',
               existing_type=sa.UUID(),
               type_=sa.String(),
               existing_nullable=True)

    # Update existing data to match employees.employee_id
    op.execute("""
        UPDATE people_analyzer pa
        SET employee_id = e.employee_id
        FROM employees e
        WHERE pa.employee_id::text = e.id::text
    """)
    op.execute("""
        UPDATE people_analyzer pa
        SET created_by = e.employee_id
        FROM employees e
        WHERE pa.created_by::text = e.id::text
    """)

    # Create new FK constraints to employees.employee_id
    op.create_foreign_key(
        'fk_people_analyzer_employee_id', 'people_analyzer', 'employees',
        ['employee_id'], ['employee_id'], ondelete='SET NULL'
    )
    op.create_foreign_key(
        'fk_people_analyzer_created_by', 'people_analyzer', 'employees',
        ['created_by'], ['employee_id'], ondelete='SET NULL'
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop new FK constraints
    op.drop_constraint('fk_people_analyzer_employee_id', 'people_analyzer', type_='foreignkey')
    op.drop_constraint('fk_people_analyzer_created_by', 'people_analyzer', type_='foreignkey')

    # Alter columns back to UUID
    op.alter_column('people_analyzer', 'employee_id',
               existing_type=sa.String(),
               type_=sa.UUID(),
               existing_nullable=True)
    op.alter_column('people_analyzer', 'created_by',
               existing_type=sa.String(),
               type_=sa.UUID(),
               existing_nullable=True)

    # Recreate original FK constraints to employees.id
    op.create_foreign_key(
        'people_analyzer_employee_id_fkey', 'people_analyzer', 'employees',
        ['employee_id'], ['id'], ondelete='SET NULL'
    )
    op.create_foreign_key(
        'people_analyzer_created_by_fkey', 'people_analyzer', 'employees',
        ['created_by'], ['id'], ondelete='SET NULL'
    )
