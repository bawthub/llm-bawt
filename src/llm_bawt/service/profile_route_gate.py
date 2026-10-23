"""A profile save may not newly put a bot on a route it can't run (TASK-1302 A1′).

A route is assignable only when it needs no credential or its owning provider
is connected (:func:`~.providers.registry.route_eligibility`). Six Claude bots
landed on the unconnected API-key route because neither the UI nor this save
checked; the bridge never noticed because it ignored the route.

The check runs inside the save's own transaction, after ``lock_target`` and on
the row about to commit, compared with the row read under that lock. A stalled
save that carries an old endpoint therefore can't write it back over a newer
one unless that old route is still eligible. A save that keeps the endpoint (a
persona or prompt edit of a bot on a dead route) is always admitted.

Provider health is probed *before* the transaction (:meth:`ProfileRouteGate.for_payload`),
never while the advisory lock is held. Harness↔path compatibility stays with
``_validate_model_endpoint`` (the authority) and the DB trigger (a coarser
backstop); this gate only judges the credential.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import text

from ..runtime_settings import BotProfile, ProfileRuntime
from .profile_switch_gate import ProfileSwitchGate
from .providers.registry import access_path_connections, route_eligibility
from .turn_admission import lock_target

_ENDPOINT_PATH_SQL = text(
    """
    SELECT a.key, a.auth_mechanism
      FROM model_endpoints e
      JOIN access_paths a ON a.id = e.access_path_id
     WHERE e.id = :endpoint_id
    """
)


class RouteIneligible(Exception):
    """The save would newly assign a route the bot can't run; nothing was written."""

    status_code = 409

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ProfileRouteGate:
    """Refuses a save that changes ``endpoint_id`` to an ineligible route.

    ``connections`` maps ``access_paths.key`` to its
    :func:`access_path_connections` entry, taken before the transaction. A path
    missing from it is refused (fail closed), never probed under the lock.
    """

    def __init__(self, connections: Mapping[str, dict[str, Any] | None]):
        self._connections = dict(connections)

    @classmethod
    def for_payload(cls, config: Any, engine: Any, payload: Mapping[str, Any]) -> "ProfileRouteGate":
        """Snapshot the connection of the endpoint the payload sets, if any.

        A staged endpoint other than the payload's is the row's own (the
        payload didn't set one), which is unchanged and never checked.
        """
        endpoint_id = payload.get("endpoint_id")
        if not isinstance(endpoint_id, int):
            return cls({})
        with engine.connect() as conn:
            path = conn.execute(_ENDPOINT_PATH_SQL, {"endpoint_id": endpoint_id}).first()
        if path is None:
            return cls({})
        return cls(access_path_connections(config, [path.key]))

    def check(self, conn: Any, previous: ProfileRuntime | None, row: BotProfile) -> None:
        endpoint_id = row.endpoint_id
        if endpoint_id is None:
            return
        if previous is not None and previous.endpoint_id == endpoint_id:
            return
        path = conn.execute(_ENDPOINT_PATH_SQL, {"endpoint_id": endpoint_id}).first()
        if path is None:
            raise RouteIneligible(f"Model endpoint {endpoint_id} no longer exists.")
        if path.key not in self._connections and path.auth_mechanism != "none":
            raise RouteIneligible(
                f"The connection for {path.key} changed during the save; try again."
            )
        verdict = route_eligibility(path.key, path.auth_mechanism, self._connections.get(path.key))
        if not verdict["eligible"]:
            raise RouteIneligible(verdict["reason"])


class ProfileSaveGuard:
    """The guard the bot-profile route save runs: one lock, then both gates.

    The system switch gate answers first, so a held runtime reports
    ``system_held``/``system_frozen`` rather than a route problem.
    """

    def __init__(self, switch: ProfileSwitchGate, route: ProfileRouteGate):
        self._switch = switch
        self._route = route

    def lock(self, conn: Any, slug: str) -> None:
        lock_target(conn, slug)

    def check(self, conn: Any, previous: ProfileRuntime | None, row: BotProfile) -> None:
        self._switch.check(conn, previous, row)
        self._route.check(conn, previous, row)
