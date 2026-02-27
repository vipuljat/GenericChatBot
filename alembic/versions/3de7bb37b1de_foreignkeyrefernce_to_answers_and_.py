"""foreignKeyReference to answers and People Analyzer

Revision ID: 3de7bb37b1de
Revises: 1ad74606718e
Create Date: 2025-12-25 18:59:19.662023
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '3de7bb37b1de'
down_revision: Union[str, Sequence[str], None] = '1ad74606718e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -------------------------------------------------------
    # ANSWERS
    # -------------------------------------------------------

    # 1. Drop old FK (VARCHAR → employees.employee_id)
    op.drop_constraint(
        op.f('answers_attempter_by_id_fkey'),
        'answers',
        type_='foreignkey'
    )

    # 2. Convert column to UUID
    op.execute("""
        ALTER TABLE answers
        ALTER COLUMN attempter_by_id
        TYPE UUID
        USING attempter_by_id::uuid;
    """)

    # 3. NULL orphaned references (THIS FIXES YOUR ERROR)
    op.execute("""
        UPDATE answers a
        SET attempter_by_id = NULL
        WHERE attempter_by_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM employees e
              WHERE e.id = a.attempter_by_id
          );
    """)

    # 4. Create correct FK → employees.id
    op.create_foreign_key(
        'answers_attempter_by_id_fkey',
        'answers',
        'employees',
        ['attempter_by_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # -------------------------------------------------------
    # PEOPLE_ANALYZER
    # -------------------------------------------------------

    op.drop_constraint(
        op.f('fk_people_analyzer_employee_id'),
        'people_analyzer',
        type_='foreignkey'
    )
    op.drop_constraint(
        op.f('fk_people_analyzer_created_by'),
        'people_analyzer',
        type_='foreignkey'
    )

    op.execute("""
        ALTER TABLE people_analyzer
        ALTER COLUMN employee_id
        TYPE UUID
        USING employee_id::uuid;
    """)

    op.execute("""
        ALTER TABLE people_analyzer
        ALTER COLUMN created_by
        TYPE UUID
        USING created_by::uuid;
    """)

    # NULL orphaned references
    op.execute("""
        UPDATE people_analyzer pa
        SET employee_id = NULL
        WHERE employee_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM employees e WHERE e.id = pa.employee_id
          );
    """)

    op.execute("""
        UPDATE people_analyzer pa
        SET created_by = NULL
        WHERE created_by IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM employees e WHERE e.id = pa.created_by
          );
    """)

    op.create_foreign_key(
        'fk_people_analyzer_employee_id',
        'people_analyzer',
        'employees',
        ['employee_id'],
        ['id'],
        ondelete='SET NULL'
    )

    op.create_foreign_key(
        'fk_people_analyzer_created_by',
        'people_analyzer',
        'employees',
        ['created_by'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    # -------------------------------------------------------
    # PEOPLE_ANALYZER (reverse UUID → VARCHAR mapped to employees.employee_id)
    # -------------------------------------------------------
    op.drop_constraint('fk_people_analyzer_created_by', 'people_analyzer', type_='foreignkey')
    op.drop_constraint('fk_people_analyzer_employee_id', 'people_analyzer', type_='foreignkey')

    op.add_column('people_analyzer', sa.Column('employee_id_str', sa.String(), nullable=True))
    op.add_column('people_analyzer', sa.Column('created_by_str', sa.String(), nullable=True))

    op.execute("""
        UPDATE people_analyzer pa
        SET employee_id_str = e.employee_id
        FROM employees e
        WHERE e.id = pa.employee_id
    """)
    op.execute("""
        UPDATE people_analyzer pa
        SET created_by_str = e.employee_id
        FROM employees e
        WHERE e.id = pa.created_by
    """)

    op.drop_column('people_analyzer', 'employee_id')
    op.drop_column('people_analyzer', 'created_by')
    op.alter_column('people_analyzer', 'employee_id_str', new_column_name='employee_id')
    op.alter_column('people_analyzer', 'created_by_str', new_column_name='created_by')

    op.create_foreign_key(
        'fk_people_analyzer_employee_id', 'people_analyzer', 'employees',
        ['employee_id'], ['employee_id'], ondelete='SET NULL'
    )
    op.create_foreign_key(
        'fk_people_analyzer_created_by', 'people_analyzer', 'employees',
        ['created_by'], ['employee_id'], ondelete='SET NULL'
    )

    # -------------------------------------------------------
    # ANSWERS (reverse UUID → VARCHAR mapped to employees.employee_id)
    # -------------------------------------------------------
    op.drop_constraint('answers_attempter_by_id_fkey', 'answers', type_='foreignkey')

    op.add_column('answers', sa.Column('attempter_by_id_str', sa.String(), nullable=True))
    op.execute("""
        UPDATE answers a
        SET attempter_by_id_str = e.employee_id
        FROM employees e
        WHERE e.id = a.attempter_by_id
    """)
    op.drop_column('answers', 'attempter_by_id')
    op.alter_column('answers', 'attempter_by_id_str', new_column_name='attempter_by_id')

    op.create_foreign_key(
        'answers_attempter_by_id_fkey', 'answers', 'employees',
        ['attempter_by_id'], ['employee_id'], ondelete='SET NULL'
    )
