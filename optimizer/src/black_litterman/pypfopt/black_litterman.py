"""
The ``black_litterman`` module houses the BlackLittermanModel class, which
generates posterior estimates of expected returns given a prior estimate and user-supplied
views. In addition, two utility functions are defined, which calculate:

- market-implied prior estimate of returns
- market-implied risk-aversion parameter
"""

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import base_optimizer
from .enums import OmegaMethod, PriorType


def market_implied_prior_returns(
    market_caps: Union[Dict[str, float], pd.Series],
    risk_aversion: float,
    cov_matrix: Union[np.ndarray, pd.DataFrame],
    risk_free_rate: float = 0.0,
) -> pd.Series:
    r"""
    Compute the prior estimate of returns implied by the market weights.

    In other words, given each asset's contribution to the risk of the market
    portfolio, how much are we expecting to be compensated?

    """
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")

    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn(
            "If cov_matrix is not a dataframe, market cap index must be aligned to cov_matrix",
            RuntimeWarning,
        )

    mcaps = pd.Series(market_caps)
    mkt_weights = mcaps / mcaps.sum()

    # Pi is excess returns so must add risk_free_rate to get absolute return
    return risk_aversion * cov_matrix.dot(mkt_weights) + risk_free_rate


class BlackLittermanModel(base_optimizer.BaseOptimizer):
    """
    Black-Litterman portfolio optimization model.

    Generates posterior estimates of expected returns by combining a prior estimate
    (typically market-implied) with investor views using Bayesian updating.

    """

    def __init__(
        self,
        cov_matrix: Union[np.ndarray, pd.DataFrame],
        pi: Optional[Union[str, np.ndarray, pd.Series]] = None,
        absolute_views: Optional[Union[Dict[str, float], pd.Series]] = None,
        Q: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        P: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        omega: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
        view_confidences: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
        tau: float = 0.05,
        risk_aversion: float = 1.0,
        **kwargs
    ) -> None:
        # Keep raw dataframes
        self._raw_cov_matrix = cov_matrix

        #  Initialise base optimizer
        if isinstance(cov_matrix, np.ndarray):
            self.cov_matrix = cov_matrix
            super().__init__(len(cov_matrix), list(range(len(cov_matrix))))
        else:
            self.cov_matrix = np.asarray(cov_matrix.values)
            super().__init__(len(cov_matrix), list(cov_matrix.columns))

        #  Sanitise inputs
        if absolute_views is not None:
            self.Q, self.P = self._parse_views(absolute_views)
        else:
            self._set_Q_P(Q, P)
        self._set_risk_aversion(risk_aversion)
        self._set_pi(pi, **kwargs)
        self._set_tau(tau)
        # Make sure all dimensions work
        self._check_attribute_dimensions()

        self._set_omega(omega, view_confidences)

        # Private intermediaries
        self._tau_sigma_P = None
        self._A = None

        self.posterior_rets = None
        self.posterior_cov = None

    def _parse_views(
        self,
        absolute_views: Union[Dict[str, float], pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a collection (dict or series) of absolute views, construct
        the appropriate views vector and picking matrix. The views must
        be a subset of the tickers in the covariance matrix.
        """
        if not isinstance(absolute_views, (dict, pd.Series)):
            raise TypeError("views should be a dict or pd.Series")
        # Coerce to series
        views = pd.Series(absolute_views)
        k = len(views)

        Q = np.zeros((k, 1))
        P = np.zeros((k, self.n_assets))

        for i, view_ticker in enumerate(views.keys()):
            try:
                Q[i] = views[view_ticker]
                P[i, list(self.tickers).index(view_ticker)] = 1
            except ValueError:
                #  Could make this smarter by just skipping
                raise ValueError("Providing a view on an asset not in the universe")
        return Q, P

    def _set_Q_P(
        self,
        Q: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]],
        P: Optional[Union[np.ndarray, pd.DataFrame]]
    ) -> None:
        if Q is None:
            raise ValueError("Q must be provided if absolute_views is not specified")

        if isinstance(Q, (pd.Series, pd.DataFrame)):
            self.Q = np.asarray(Q.values).reshape(-1, 1)
        elif isinstance(Q, np.ndarray):
            self.Q = Q.reshape(-1, 1)
        else:
            raise TypeError("Q must be an array or dataframe")

        if isinstance(P, pd.DataFrame):
            self.P = np.asarray(P.values)
        elif isinstance(P, np.ndarray):
            self.P = P
        elif len(self.Q) == self.n_assets:
            # If a view on every asset is provided, P defaults
            # to the identity matrix.
            self.P = np.eye(self.n_assets)
        else:
            raise TypeError("P must be an array or dataframe")

    def _set_pi(
        self,
        pi: Optional[Union[str, np.ndarray, pd.Series, pd.DataFrame]],
        **kwargs
    ) -> None:
        if pi is None:
            warnings.warn("Running Black-Litterman with no prior.")
            self.pi = np.zeros((self.n_assets, 1))
        elif isinstance(pi, (pd.Series, pd.DataFrame)):
            self.pi = np.asarray(pi.values).reshape(-1, 1)
        elif isinstance(pi, np.ndarray):
            self.pi = pi.reshape(-1, 1)
        elif pi == PriorType.MARKET or pi == "market":
            if "market_caps" not in kwargs:
                raise ValueError(
                    "Please pass a series/array of market caps via the market_caps keyword argument"
                )
            # We've validated market_caps exists, so use direct access
            market_caps = kwargs["market_caps"]
            risk_free_rate = kwargs.get("risk_free_rate", 0.0)

            market_prior = market_implied_prior_returns(
                market_caps, self.risk_aversion, self._raw_cov_matrix, risk_free_rate
            )
            self.pi = np.asarray(market_prior.values).reshape(-1, 1)
        elif pi == PriorType.EQUAL or pi == "equal":
            self.pi = np.ones((self.n_assets, 1)) / self.n_assets
        else:
            raise TypeError("pi must be an array, series, 'market', or 'equal'")

    def _set_tau(self, tau: float) -> None:
        if tau <= 0 or tau > 1:
            raise ValueError("tau should be between 0 and 1")
        self.tau = tau

    def _set_risk_aversion(self, risk_aversion: float) -> None:
        if risk_aversion <= 0:
            raise ValueError("risk_aversion should be a positive float")
        self.risk_aversion = risk_aversion

    def _set_omega(
        self,
        omega: Optional[Union[str, np.ndarray, pd.DataFrame]],
        view_confidences: Optional[Union[np.ndarray, pd.Series, List[float]]]
    ) -> None:
        if isinstance(omega, pd.DataFrame):
            self.omega = np.asarray(omega.values)
        elif isinstance(omega, np.ndarray):
            self.omega = omega
        elif omega == OmegaMethod.IDZOREK or omega == "idzorek":
            if view_confidences is None:
                raise ValueError(
                    "To use Idzorek's method, please supply a vector of percentage "
                    "confidence levels for each view."
                )
            if not isinstance(view_confidences, np.ndarray):
                try:
                    view_confidences = np.array(view_confidences).reshape(-1, 1)
                    assert view_confidences.shape[0] == self.Q.shape[0]
                    assert np.issubdtype(view_confidences.dtype, np.number)
                except AssertionError:
                    raise ValueError(
                        "view_confidences should be a numpy 1D array or vector with the same length "
                        "as the number of views."
                    )

            self.omega = BlackLittermanModel.idzorek_method(
                view_confidences,
                self.cov_matrix,
                self.pi,
                self.Q,
                self.P,
                self.tau,
                self.risk_aversion,
            )
        elif omega is None or omega == OmegaMethod.DEFAULT or omega == "default":
            self.omega = BlackLittermanModel.default_omega(
                self.cov_matrix, self.P, self.tau
            )
        else:
            raise TypeError("omega must be a square array, dataframe, 'default', or 'idzorek'")

        K = len(self.Q)
        assert self.omega.shape == (K, K), "omega must have dimensions KxK"

    def _check_attribute_dimensions(self) -> None:
        """
        Helper method to ensure that all of the attributes created by the initialiser
        have the correct dimensions, to avoid linear algebra errors later on.
        """
        N = self.n_assets
        K = len(self.Q)
        assert self.pi.shape == (N, 1), "pi must have dimensions Nx1"
        assert self.P.shape == (K, N), "P must have dimensions KxN"
        assert self.cov_matrix.shape == (N, N), "cov_matrix must have shape NxN"

    @staticmethod
    def default_omega(
        cov_matrix: np.ndarray,
        P: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """
        Calculate the view uncertainty matrix using the method of He and Litterman (1999).

        The ratio omega/tau is proportional to the variance of the view portfolio.

        :param cov_matrix: NxN covariance matrix
        :type cov_matrix: np.ndarray
        :param P: KxN picking matrix
        :type P: np.ndarray
        :param tau: weight-on-views scalar
        :type tau: float
        :return: KxK diagonal uncertainty matrix
        :rtype: np.ndarray
        """
        return np.diag(np.diag(tau * P @ cov_matrix @ P.T))

    @staticmethod
    def idzorek_method(
        view_confidences: np.ndarray,
        cov_matrix: np.ndarray,
        pi: np.ndarray,
        Q: np.ndarray,
        P: np.ndarray,
        tau: float,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """
        Create the uncertainty matrix using Idzorek's method with user-specified confidences.

        Uses the closed-form solution described by Jay Walters in
        The Black-Litterman Model in Detail (2014).
        """
        view_omegas = []
        for view_idx in range(len(Q)):
            conf = view_confidences[view_idx]

            if conf < 0 or conf > 1:
                raise ValueError("View confidences must be between 0 and 1")

            # Special handler to avoid dividing by zero.
            # If zero conf, return very big number as uncertainty
            if conf == 0:
                view_omegas.append(1e6)
                continue

            P_view = P[view_idx].reshape(1, -1)
            alpha = (1 - conf) / conf  # formula (44)
            omega = tau * alpha * P_view @ cov_matrix @ P_view.T  # formula (41)
            view_omegas.append(omega.item())

        return np.diag(view_omegas)

    def bl_returns(self) -> pd.Series:
        """
        Calculate the posterior estimate of the returns vector,
        given views on some assets.

        Uses Bayesian updating to blend prior returns with user views.
        """

        if self._tau_sigma_P is None:
            self._tau_sigma_P = self.tau * self.cov_matrix @ self.P.T

        # Solve the linear system Ax = b to avoid inversion
        if self._A is None:
            self._A = (self.P @ self._tau_sigma_P) + self.omega
        b = self.Q - self.P @ self.pi
        try:
            solution = np.linalg.solve(self._A, b)
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                solution = np.linalg.lstsq(self._A, b, rcond=None)[0]
            else:
                raise e
        post_rets = self.pi + self._tau_sigma_P @ solution
        return pd.Series(post_rets.flatten(), index=self.tickers)

    def bl_cov(self) -> pd.DataFrame:
        """
        Calculate the posterior estimate of the covariance matrix,
        given views on some assets.

        Based on He and Litterman (2002). Assumes omega is diagonal.
        If this is not the case, please manually set omega_inv.
        """
        if self._tau_sigma_P is None:
            self._tau_sigma_P = self.tau * self.cov_matrix @ self.P.T
        if self._A is None:
            self._A = (self.P @ self._tau_sigma_P) + self.omega

        b = self._tau_sigma_P.T
        try:
            M_solution = np.linalg.solve(self._A, b)
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                M_solution = np.linalg.lstsq(self._A, b, rcond=None)[0]
            else:
                raise e
        M = self.tau * self.cov_matrix - self._tau_sigma_P @ M_solution
        posterior_cov = self.cov_matrix + M
        return pd.DataFrame(posterior_cov, index=self.tickers, columns=self.tickers)

    def bl_weights(
        self,
        risk_aversion: Optional[float] = None
    ) -> 'OrderedDict':
        r"""
        Compute the optimal portfolio weights implied by the posterior returns.

        Uses the formula: w = (δΣ)^{-1} E(R)

        This is a special case of mean-variance optimization.
        """
        if risk_aversion is None:
            risk_aversion = self.risk_aversion

        self.posterior_rets = self.bl_returns()
        A = risk_aversion * self.cov_matrix
        b = self.posterior_rets
        try:
            weight_solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                weight_solution = np.linalg.lstsq(A, b, rcond=None)[0]
            else:
                raise e
        raw_weights = weight_solution
        self.weights = raw_weights / raw_weights.sum()
        return self._make_output_weights()

    def optimize(
        self,
        risk_aversion: Optional[float] = None
    ) -> 'OrderedDict':
        """
        Alias for bl_weights for consistency with other optimization methods.

        """
        return self.bl_weights(risk_aversion)

    def portfolio_performance(
        self,
        verbose: bool = False,
        risk_free_rate: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Calculate (and optionally print) the performance of the optimal portfolio.

        Uses the Black-Litterman posterior returns and covariance matrix.
        Calculates expected return, volatility, and the Sharpe ratio.
        """
        if self.weights is None:
            raise ValueError("Weights not yet computed. Call bl_weights() or optimize() first.")
        if self.posterior_rets is None:
            raise ValueError("Posterior returns not computed. This should not happen after bl_weights().")
        if self.posterior_cov is None:
            self.posterior_cov = self.bl_cov()

        # base_optimizer.portfolio_performance returns tuple[float | None, float, float | None]
        # but in BL case with posterior_rets, all values are guaranteed to be float
        mu, sigma, sharpe = base_optimizer.portfolio_performance(
            self.weights,
            self.posterior_rets,
            self.posterior_cov,
            verbose,
            risk_free_rate,
        )

        # Assert non-None for type checker (guaranteed in BL case with posterior returns)
        assert mu is not None, "Expected return should not be None with posterior returns"
        assert sharpe is not None, "Sharpe ratio should not be None with posterior returns"

        return (mu, sigma, sharpe)
