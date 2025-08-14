"""Initial database

Revision ID: 3dbf59917687
Revises: 
Create Date: 2025-08-14 15:10:54.554903

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3dbf59917687'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # 1. Employees table (independent)
    op.create_table(
        'employees',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('first_name', sa.String(length=50), nullable=False),
        sa.Column('last_name', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=True),
        sa.Column('hire_date', sa.Date(), nullable=False),
        sa.Column('job_title', sa.String(length=100), nullable=False),
        sa.Column('salary', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_employees_email'), 'employees', ['email'], unique=True)
    op.create_index(op.f('ix_employees_id'), 'employees', ['id'], unique=False)

    # 2. Users table (depends on employees)
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=True),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='TRUE', nullable=False),
        sa.Column('is_superuser', sa.Boolean(), server_default='FALSE', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('role_id', sa.Integer(), nullable=True),
        sa.Column('employee_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('employee_id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=False)

    # 3. Roles table (depends on users)
    op.create_table(
        'roles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('code', sa.String(length=50), nullable=True),
        sa.Column('created_by_user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_roles_id'), 'roles', ['id'], unique=False)

    # Add missing foreign key for role_id in users (now that roles exists)
    op.create_foreign_key(
        'fk_users_role_id_roles',
        'users', 'roles',
        ['role_id'], ['id']
    )

    # 4. Image categories (depends on users)
    op.create_table(
        'imagecategories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('created_by_user_id', sa.Integer(), nullable=True),
        sa.Column('updated_by_user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['updated_by_user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_imagecategories_id'), 'imagecategories', ['id'], unique=False)

    # 5. Permissions (depends on users)
    op.create_table(
        'permissions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('code', sa.String(length=50), nullable=False),
        sa.Column('module_name', sa.String(length=50), nullable=True),
        sa.Column('created_by_user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_permissions_id'), 'permissions', ['id'], unique=False)

    # 6. User-Role link table
    op.create_table(
        'user_role',
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('role_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'])
    )

    # 7. Images (depends on imagecategories and users)
    op.create_table(
        'images',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=30), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('image_path', sa.String(length=255), nullable=False),
        sa.Column('upload_date', sa.DateTime(), nullable=False),
        sa.Column('category_id', sa.Integer(), nullable=True),
        sa.Column('created_by_user_id', sa.Integer(), nullable=True),
        sa.Column('updated_by_user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['category_id'], ['imagecategories.id']),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['updated_by_user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_images_id'), 'images', ['id'], unique=False)

    # 8. Role-Permission link table
    op.create_table(
        'role_permission',
        sa.Column('role_id', sa.Integer(), nullable=True),
        sa.Column('permission_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.id']),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'])
    )

    # 9. User-Permission link table
    op.create_table(
        'user_permission',
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('permission_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'])
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('user_permission')
    op.drop_table('role_permission')
    op.drop_index(op.f('ix_images_id'), table_name='images')
    op.drop_table('images')
    op.drop_table('user_role')
    op.drop_index(op.f('ix_permissions_id'), table_name='permissions')
    op.drop_table('permissions')
    op.drop_index(op.f('ix_imagecategories_id'), table_name='imagecategories')
    op.drop_table('imagecategories')
    op.drop_index(op.f('ix_roles_id'), table_name='roles')
    op.drop_table('roles')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_employees_id'), table_name='employees')
    op.drop_index(op.f('ix_employees_email'), table_name='employees')
    op.drop_table('employees')
