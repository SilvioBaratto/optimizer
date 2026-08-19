"""Shared base schemas for API response models."""

import uuid
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    """Base model that serializes field names to camelCase.

    All response models returned to the Angular frontend must extend this class
    to ensure consistent camelCase JSON serialization.
    Request models that are not exposed to the frontend may use plain BaseModel.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


def _coerce_uuid(v: Any) -> str:
    """Coerce uuid.UUID to str for Pydantic v2 from_attributes mode."""
    if isinstance(v, uuid.UUID):
        return str(v)
    return str(v)


StrFromUUID = Annotated[str, BeforeValidator(_coerce_uuid)]
