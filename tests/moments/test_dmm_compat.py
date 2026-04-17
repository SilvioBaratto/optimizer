"""Tests for DMM import guard utility (_dmm_compat)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestHasDmm:
    def test_has_dmm_is_bool(self) -> None:
        from optimizer.moments._dmm_compat import HAS_DMM

        assert isinstance(HAS_DMM, bool)

    def test_has_dmm_true_when_available(self) -> None:
        """When torch/pyro are importable, HAS_DMM reflects availability."""
        from optimizer.moments._dmm_compat import HAS_DMM

        try:
            import pyro  # noqa: F401
            import torch  # noqa: F401

            expected = True
        except ImportError:
            expected = False

        assert HAS_DMM is expected


class TestRequireDmm:
    def test_require_dmm_succeeds_when_available(self) -> None:
        """When HAS_DMM is True, require_dmm() returns None without raising."""
        from optimizer.moments._dmm_compat import require_dmm

        with patch("optimizer.moments._dmm_compat.HAS_DMM", True):
            result = require_dmm()
            assert result is None

    def test_require_dmm_raises_when_unavailable(self) -> None:
        """When HAS_DMM is False, require_dmm() raises ImportError."""
        from optimizer.moments._dmm_compat import require_dmm

        with (
            patch("optimizer.moments._dmm_compat.HAS_DMM", False),
            pytest.raises(ImportError),
        ):
            require_dmm()

    def test_require_dmm_error_message_contains_install_hint(self) -> None:
        """Error message must contain the pip install hint."""
        from optimizer.moments._dmm_compat import require_dmm

        with (
            patch("optimizer.moments._dmm_compat.HAS_DMM", False),
            pytest.raises(ImportError, match='pip install "portopt\\[dmm\\]"'),
        ):
            require_dmm()


class TestDmmModuleImportable:
    def test_dmm_module_importable_without_runtime_error(self) -> None:
        """_dmm.py must be importable even when HAS_DMM is False."""
        import importlib

        import optimizer.moments._dmm as dmm_module

        importlib.reload(dmm_module)  # ensure fresh import path is exercised

    def test_dmm_config_importable_without_torch(self) -> None:
        """DMMConfig must be accessible regardless of torch/pyro availability."""
        from optimizer.moments._dmm import DMMConfig

        cfg = DMMConfig()
        assert cfg.z_dim == 16

    def test_dmm_result_importable_without_torch(self) -> None:
        """DMMResult class must be importable regardless of torch/pyro availability."""
        from optimizer.moments._dmm import DMMResult

        assert DMMResult is not None

    def test_fit_dmm_raises_import_error_when_dmm_unavailable(self) -> None:
        """fit_dmm() must raise ImportError with hint when DMM deps absent."""
        import numpy as np
        import pandas as pd

        from optimizer.moments._dmm import fit_dmm

        dummy = pd.DataFrame(
            np.zeros((10, 2)),
            columns=["A", "B"],
        )

        with (
            patch("optimizer.moments._dmm_compat.HAS_DMM", False),
            pytest.raises(ImportError, match='pip install "portopt\\[dmm\\]"'),
        ):
            fit_dmm(dummy)

    def test_blend_moments_dmm_raises_import_error_when_dmm_unavailable(
        self,
    ) -> None:
        """blend_moments_dmm() must raise ImportError with hint when DMM deps absent."""
        import numpy as np
        import pandas as pd

        from optimizer.moments._dmm import DMMResult, blend_moments_dmm

        dummy_result = DMMResult(
            latent_means=pd.DataFrame(),
            latent_stds=pd.DataFrame(),
            elbo_history=[],
            model=None,
            tickers=[],
            input_mean=np.array([]),
            input_std=np.array([]),
        )

        with (
            patch("optimizer.moments._dmm_compat.HAS_DMM", False),
            pytest.raises(ImportError, match='pip install "portopt\\[dmm\\]"'),
        ):
            blend_moments_dmm(dummy_result)


class TestInitImportsDmmUnconditionally:
    def test_dmm_symbols_accessible_from_moments_package(self) -> None:
        """DMM symbols must be importable from optimizer.moments."""
        from optimizer.moments import DMMConfig, DMMResult, blend_moments_dmm, fit_dmm

        assert DMMConfig is not None
        assert DMMResult is not None
        assert callable(fit_dmm)
        assert callable(blend_moments_dmm)

    def test_moments_init_has_no_contextlib_suppress(self) -> None:
        """__init__.py must not use contextlib.suppress for DMM imports."""
        import inspect

        import optimizer.moments as moments_pkg

        source = inspect.getsource(moments_pkg)
        assert "contextlib.suppress" not in source
