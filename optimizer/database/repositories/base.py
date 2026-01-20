"""
Base Repository - Common functionality for all repositories.
"""

import logging
from typing import TypeVar, Generic, Type, Optional, List, Any
from contextlib import contextmanager

from sqlalchemy.orm import Session

from optimizer.database.database import DatabaseManager

logger = logging.getLogger(__name__)

# Type variable for SQLAlchemy models
T = TypeVar("T")


class BaseRepository(Generic[T]):
    """
    Base class for all repository implementations.
    """

    def __init__(self, db_manager: DatabaseManager, model_class: Type[T]):
        """
        Initialize repository.
        """
        self._db_manager = db_manager
        self._model_class = model_class
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @contextmanager
    def _get_session(self):
        """
        Get database session with automatic cleanup.
        """
        with self._db_manager.get_session() as session:
            yield session

    def _get_by_id(self, id: Any) -> Optional[T]:
        """
        Get a single record by primary key.
        """
        with self._get_session() as session:
            return session.get(self._model_class, id)

    def _get_all(self, limit: int = 1000) -> List[T]:
        """
        Get all records with optional limit.
        """
        with self._get_session() as session:
            return session.query(self._model_class).limit(limit).all()

    def _count(self) -> int:
        """
        Count total records.
        """
        with self._get_session() as session:
            return session.query(self._model_class).count()

    def _save(self, instance: T) -> T:
        """
        Save a model instance.
        """
        with self._get_session() as session:
            session.add(instance)
            session.commit()
            session.refresh(instance)
            return instance

    def _save_batch(self, instances: List[T]) -> List[T]:
        """
        Save multiple model instances.
        """
        with self._get_session() as session:
            session.add_all(instances)
            session.commit()
            for instance in instances:
                session.refresh(instance)
            return instances

    def _delete(self, instance: T) -> None:
        """
        Delete a model instance.
        """
        with self._get_session() as session:
            session.delete(instance)
            session.commit()

    def _execute_query(self, query_func):
        """
        Execute a query function within a session.
        """
        with self._get_session() as session:
            return query_func(session)
