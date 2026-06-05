"""Unit tests for the shared API assertion helpers (issue #807).

Covers the pass and fail path of each helper: ``assert_json_safe``,
``assert_response_matches_schema``, ``assert_camel_case_keys`` and
``assert_no_snake_case_keys``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from tests._fixtures import (
    assert_camel_case_keys,
    assert_json_safe,
    assert_no_snake_case_keys,
    assert_response_matches_schema,
)


class _SampleSchema(BaseModel):
    name: str
    value: int


class TestAssertJsonSafe:
    def test_when_plain_json_payload_passed_no_error_is_raised(self):
        assert_json_safe({"a": 1, "b": [1.5, "x", True, None], "c": {"d": 2}})

    def test_when_numpy_float_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_json_safe({"x": np.float64(1.0)})

    def test_when_numpy_array_nested_in_list_path_is_reported(self):
        with pytest.raises(AssertionError, match=r"\[1\]"):
            assert_json_safe({"weights": ["ok", np.array([1, 2, 3])]})

    def test_when_decimal_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_json_safe({"price": Decimal("1.23")})

    def test_when_datetime_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_json_safe({"ts": datetime.now(timezone.utc)})

    def test_when_dataframe_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_json_safe({"df": pd.DataFrame({"a": [1, 2]})})

    def test_when_nan_float_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_json_safe({"x": float("nan")})

    def test_when_inf_float_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_json_safe({"x": float("inf")})

    def test_when_tuple_payload_passed_no_error_is_raised(self):
        # Tuples serialise to JSON arrays via json.dumps — accepted by design.
        assert_json_safe({"weights": (0.6, 0.4)})


class TestAssertResponseMatchesSchema:
    def test_when_valid_payload_passed_validated_model_is_returned(self):
        model = assert_response_matches_schema(
            {"name": "ok", "value": 3}, _SampleSchema
        )
        assert isinstance(model, _SampleSchema)
        assert model.value == 3

    def test_when_invalid_payload_passed_assertion_error_names_schema(self):
        with pytest.raises(AssertionError, match="_SampleSchema"):
            assert_response_matches_schema(
                {"name": "ok", "value": "NaN"}, _SampleSchema
            )


class TestAssertCamelCaseKeys:
    def test_when_all_keys_camel_case_no_error_is_raised(self):
        assert_camel_case_keys({"jobId": 1, "nested": {"riskScore": 2}})

    def test_when_snake_case_key_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_camel_case_keys({"job_id": 1})

    def test_when_snake_case_key_nested_in_list_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_camel_case_keys({"items": [{"snake_key": 1}]})


class TestAssertNoSnakeCaseKeys:
    def test_when_no_underscore_keys_no_error_is_raised(self):
        assert_no_snake_case_keys({"jobId": 1, "nested": {"riskScore": 2}})

    def test_when_underscore_key_present_assertion_error_is_raised(self):
        with pytest.raises(AssertionError):
            assert_no_snake_case_keys({"job_id": 1})

    def test_when_underscore_key_in_allow_set_no_error_is_raised(self):
        assert_no_snake_case_keys({"job_id": 1}, allow=frozenset({"job_id"}))
