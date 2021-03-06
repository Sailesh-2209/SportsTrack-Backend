"""empty message

Revision ID: 6ffd2d7c7b99
Revises: c66798a82de1
Create Date: 2022-06-21 06:19:06.185681

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6ffd2d7c7b99'
down_revision = 'c66798a82de1'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('sessions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=True),
    sa.Column('duration', sa.Integer(), nullable=True),
    sa.Column('vidDir', sa.String(length=64), nullable=True),
    sa.Column('heatmapPath', sa.String(length=64), nullable=True),
    sa.Column('pieChartPath', sa.String(length=64), nullable=True),
    sa.Column('heatmapURL', sa.String(length=64), nullable=True),
    sa.Column('pieChartURL', sa.String(length=64), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('heatmapPath'),
    sa.UniqueConstraint('heatmapURL'),
    sa.UniqueConstraint('name'),
    sa.UniqueConstraint('pieChartPath'),
    sa.UniqueConstraint('pieChartURL'),
    sa.UniqueConstraint('vidDir')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('sessions')
    # ### end Alembic commands ###
