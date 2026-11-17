"""App-owned removal reservations, serialized with every house assignment."""
from __future__ import annotations

from typing import Any

from sqlalchemy import Column, MetaData, String, Table, select

from .bot_profile_changes import ProfileWriteConflict

_metadata = MetaData()
RESERVATIONS = Table(
    "house_retirements", _metadata,
    Column("house", String(32), primary_key=True),
    Column("reconcile_id", String(64), nullable=False),
)


class HouseAssignmentRetiring(ProfileWriteConflict):
    code = "house_retiring"


def ensure_schema(conn: Any) -> None:
    _metadata.create_all(conn)
    if conn.dialect.name == "postgresql":
        from .bot_tags.catalog import require_owned
        require_owned(conn, "house_retirements")


def reservations(conn: Any) -> dict[str, str]:
    """The caller's transaction sees reservations; failures never mean free."""
    return {house: reconcile_id for house, reconcile_id in conn.execute(select(
        RESERVATIONS.c.house, RESERVATIONS.c.reconcile_id,
    ))}


def assert_assignable(conn: Any, house: str | None, slug: str) -> None:
    """Call only under house_tenancy's assignment lock, before any write."""
    from agent_bridge.houses import normalize_house
    house = normalize_house(house)
    if house and house in reservations(conn):
        raise HouseAssignmentRetiring(
            slug, {"retiring_house": house},
            f"{house} is reserved for removal; choose another house",
        )
