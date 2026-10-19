from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(scope="session")
def _postgres_container() -> Iterator[Any]:
    required = os.environ.get("BBA_REQUIRE_POSTGRES_TESTS") == "1"
    try:
        import docker
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        if required:
            pytest.fail("Required PostgreSQL test dependencies are unavailable")
        pytest.skip("PostgreSQL test dependencies are unavailable")

    client = None
    try:
        client = docker.from_env()
        client.ping()
    except docker.errors.DockerException:
        if required:
            pytest.fail("Required PostgreSQL tests cannot connect to Docker")
        pytest.skip("Docker is unavailable for PostgreSQL tests")
    finally:
        if client is not None:
            client.close()

    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres
