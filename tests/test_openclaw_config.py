"""OpenClaw bridge runtime configuration checks."""

from sqlalchemy import create_engine

from openclaw_bridge.config import BridgeConfig


def test_bridge_uses_driver_installed_in_slim_image() -> None:
    config = BridgeConfig(
        postgres_user="test", postgres_password="unused", postgres_database="test"
    )
    # Engine construction imports the selected DBAPI without opening a connection.
    engine = create_engine(config.postgres_url)
    try:
        assert engine.dialect.driver == "psycopg2"
    finally:
        engine.dispose()
