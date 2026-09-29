"""Shared base schemas for API response models."""

import uuid
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    """Base model with camelCase JSON serialization.

    Extend for any response schema that must serialize field names to camelCase.
    Internal service schemas that are never serialized to JSON may use plain
    ``BaseModel`` instead.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


def _coerce_uuid(v: Any) -> str:
    if isinstance(v, uuid.UUID):
        return str(v)
    return str(v)


StrFromUUID = Annotated[str, BeforeValidator(_coerce_uuid)]
