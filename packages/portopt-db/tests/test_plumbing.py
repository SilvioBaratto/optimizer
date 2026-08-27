"""P1 plumbing unit tests: Base registry, DbConfig, RepositoryBase._upsert guards."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from portopt_db.base import Base, BaseModel
from portopt_db.config import DbConfig
from portopt_db.repository import RepositoryBase


class TestBase:
    def test_base_and_basemodel_share_metadata(self) -> None:
        assert BaseModel.metadata is Base.metadata

    def test_basemodel_is_abstract(self) -> None:
        assert BaseModel.__abstract__ is True


class TestDbConfig:
    def test_defaults_and_connect_args(self) -> None:
        cfg = DbConfig(url="postgresql://u:p@h:5432/d", application_name="svc")
        assert cfg.pool_pre_ping is True
        ca = cfg.connect_args()
        assert ca["application_name"] == "svc"
        assert ca["keepalives"] == 1

    def test_is_frozen(self) -> None:
        cfg = DbConfig(url="x")
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            cfg.url = "y"  # type: ignore[misc]


class TestUpsertGuards:
    def _repo(self) -> RepositoryBase:
        return RepositoryBase(MagicMock(name="session"))

    def test_empty_rows_short_circuits(self) -> None:
        assert self._repo()._upsert(MagicMock(), [], constraint_name="c") == 0

    def test_requires_exactly_one_conflict_target(self) -> None:
        repo = self._repo()
        rows = [{"a": 1}]
        with pytest.raises(ValueError, match="exactly one"):
            repo._upsert(MagicMock(), rows)  # neither
        with pytest.raises(ValueError, match="exactly one"):
            repo._upsert(MagicMock(), rows, constraint_name="c", index_elements=["a"])
