"""add total_conversations and last_active to user_profiles

Revision ID: 371b34ab659f
Revises:
Create Date: 2026-02-04 12:54:02.777897

"""

from datetime import datetime
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "371b34ab659f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add columns as NULLABLE first
    op.add_column(
        "user_profiles", sa.Column("total_conversations", sa.Integer(), nullable=True)
    )
    op.add_column(
        "user_profiles", sa.Column("last_active", sa.DateTime(), nullable=True)
    )

    # Update existing rows with default values
    op.execute(
        "UPDATE user_profiles SET total_conversations = 0 WHERE total_conversations IS NULL"
    )
    op.execute(
        f"UPDATE user_profiles SET last_active = '{datetime.utcnow().isoformat()}' WHERE last_active IS NULL"
    )

    # Now make them NOT NULL (SQLite limitation workaround)
    # For SQLite, we have to recreate the table
    with op.batch_alter_table("user_profiles") as batch_op:
        batch_op.alter_column("total_conversations", nullable=False)
        # Keep last_active as nullable since it's nullable in your model


def downgrade():
    op.drop_column("user_profiles", "last_active")
    op.drop_column("user_profiles", "total_conversations")
