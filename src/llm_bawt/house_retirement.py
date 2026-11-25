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
    if conn.dialect.name == "postgresql":
        from .bot_tags.catalog import owned, require_owned
        # SQLAlchemy's unqualified checkfirst can see a later search_path
        # schema's table. Check ownership, then create in current_schema().
        if not owned(conn, "house_retirements"):
            RESERVATIONS.create(conn)
        require_owned(conn, "house_retirements")
    else:
        _metadata.create_all(conn)


_guard = None


def ensure_ready(engine: Any) -> None:
    """Create house_retirements if missing, without other profile DDL.

    A ready tenant takes no locks beyond a catalog read. A missing table is
    created under the shared schema-bootstrap advisory lock, so it serializes
    with the app's profile-store bootstrap (TASK-2637)."""
    global _guard
    from sqlalchemy import inspect

    with engine.connect() as conn:
        if conn.dialect.name == "postgresql":
            from .bot_tags.catalog import owned
            if owned(conn, "house_retirements"):
                return
        elif inspect(conn).has_table("house_retirements"):
            return
    if _guard is None:
        from .utils.schema import SchemaBootstrapGuard
        _guard = SchemaBootstrapGuard()
    _guard.run(engine, "house-retirements", ensure_schema)


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
