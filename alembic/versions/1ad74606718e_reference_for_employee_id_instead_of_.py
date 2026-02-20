from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '1ad74606718e'
down_revision = '2c8b0dae2ab5'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Drop existing FK
    op.drop_constraint(op.f('chatbot_access_employee_id_fkey'), 'chatbot_access', type_='foreignkey')

    # 2. Create a temporary UUID column
    op.add_column('chatbot_access', sa.Column('employee_uuid', sa.dialects.postgresql.UUID(), nullable=True))

    # 3. Populate the new UUID column from employees.id
    op.execute("""
        UPDATE chatbot_access
        SET employee_uuid = e.id
        FROM employees e
        WHERE chatbot_access.employee_id = e.employee_id
    """)

    # 4. Drop old string column
    op.drop_column('chatbot_access', 'employee_id')

    # 5. Rename temp UUID column to employee_id
    op.alter_column('chatbot_access', 'employee_uuid', new_column_name='employee_id')

    # 6. Add FK to employees.id
    op.create_foreign_key(
        None,
        'chatbot_access',
        'employees',
        ['employee_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # 7. Add employee_code column if not exists
    conn = op.get_bind()
    result = conn.execute(sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name='employees' AND column_name='employee_code'
    """)).fetchone()

    if not result:
        op.add_column('employees', sa.Column('employee_code', sa.String(), nullable=True))

    # 8. Populate employee_code with employee_id
    op.execute("""
        UPDATE employees
        SET employee_code = employee_id
    """)


def downgrade():
    # Reverse the upgrade (drop employee_code, revert employee_id to string)
    op.drop_constraint(None, 'chatbot_access', type_='foreignkey')
    op.add_column('chatbot_access', sa.Column('employee_str', sa.String(), nullable=True))
    op.execute("""
        UPDATE chatbot_access
        SET employee_str = e.employee_id
        FROM employees e
        WHERE chatbot_access.employee_id = e.id
    """)
    op.drop_column('chatbot_access', 'employee_id')
    op.alter_column('chatbot_access', 'employee_str', new_column_name='employee_id')

    # Re-add old FK to employee_id
    op.create_foreign_key(
        op.f('chatbot_access_employee_id_fkey'),
        'chatbot_access',
        'employees',
        ['employee_id'],
        ['employee_id'],
        ondelete='SET NULL'
    )

    # Drop employee_code column
    op.drop_column('employees', 'employee_code')
