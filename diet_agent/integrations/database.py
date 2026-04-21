"""Database integration adapters."""


def PostgreSQLClient(*args, **kwargs):
    from src.database.postgres_client import PostgreSQLClient as _PostgreSQLClient
    return _PostgreSQLClient(*args, **kwargs)


def get_postgres_client(*args, **kwargs):
    from src.database.postgres_client import get_postgres_client as _get_postgres_client
    return _get_postgres_client(*args, **kwargs)


__all__ = ["PostgreSQLClient", "get_postgres_client"]
