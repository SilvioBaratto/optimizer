"""Deep Markov Model for regime-conditional moment estimation.

Implements the architecture from Krishnan et al. (2016) "Structured Inference
Networks for Nonlinear State Space Models" using Pyro SVI with KL annealing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

# ---------------------------------------------------------------------------
# Config and Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DMMConfig:
    """Hyper-parameters for the Deep Markov Model.

    Attributes
    ----------
    z_dim : int
        Dimension of the continuous latent state z_t.
    emission_dim : int
        Hidden layer size of the Emitter MLP.
    transition_dim : int
        Hidden layer size of the GatedTransition MLP.
    rnn_dim : int
        GRU hidden state size for the backward inference network.
    num_epochs : int
        Number of SVI training epochs.
    learning_rate : float
        Initial learning rate for ClippedAdam.
    annealing_epochs : int
        Epochs over which the KL weight is linearly annealed from
        *minimum_annealing_factor* to 1.0.
    minimum_annealing_factor : float
        Starting KL annealing weight (prevents posterior collapse).
    random_state : int or None
        Seed for reproducible initialisation.
    """

    z_dim: int = 16
    emission_dim: int = 64
    transition_dim: int = 64
    rnn_dim: int = 128
    num_epochs: int = 1000
    learning_rate: float = 3e-4
    annealing_epochs: int = 50
    minimum_annealing_factor: float = 0.2
    random_state: int | None = None


@dataclass
class DMMResult:
    """Result of fitting a Deep Markov Model.

    Attributes
    ----------
    latent_means : pd.DataFrame, shape (T, z_dim)
        Variational posterior means for each time step.
    latent_stds : pd.DataFrame, shape (T, z_dim)
        Variational posterior standard deviations.
    elbo_history : list[float]
        ELBO value (= −SVI loss) per training epoch.
    model : Any
        Trained DMM nn.Module instance.
    tickers : list[str]
        Asset names, in training order.
    input_mean : ndarray, shape (n_assets,)
        Per-asset mean used for input standardisation.
    input_std : ndarray, shape (n_assets,)
        Per-asset std used for input standardisation.
    """

    latent_means: pd.DataFrame
    latent_stds: pd.DataFrame
    elbo_history: list[float]
    model: Any
    tickers: list[str]
    input_mean: npt.NDArray[np.float64]
    input_std: npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------


class Emitter(nn.Module):
    """Maps latent z_t to (loc, scale) of the observed return distribution."""

    def __init__(self, input_dim: int, z_dim: int, emission_dim: int) -> None:
        super().__init__()
        self.lin_z_h1 = nn.Linear(z_dim, emission_dim)
        self.lin_h1_h2 = nn.Linear(emission_dim, emission_dim)
        self.lin_h2_loc = nn.Linear(emission_dim, input_dim)
        self.lin_h2_scale = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.relu(self.lin_z_h1(z_t))
        h2 = self.relu(self.lin_h1_h2(h1))
        loc = self.lin_h2_loc(h2)
        scale = self.softplus(self.lin_h2_scale(h2)) + 1e-5
        return loc, scale


class GatedTransition(nn.Module):
    """Gated transition p(z_t | z_{t-1}) with learned interpolation gate.

    The gate interpolates between a linear residual (identity-initialised)
    and a nonlinear proposed mean, giving stable gradients at initialisation.
    """

    def __init__(self, z_dim: int, transition_dim: int) -> None:
        super().__init__()
        self.lin_gate_zh = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hz = nn.Linear(transition_dim, z_dim)
        self.lin_prop_zh = nn.Linear(z_dim, transition_dim)
        self.lin_prop_hz = nn.Linear(transition_dim, z_dim)
        self.lin_scale = nn.Linear(z_dim, z_dim)
        self.lin_residual = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.lin_residual.weight)
        nn.init.zeros_(self.lin_residual.bias)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self, z_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(
            self.lin_gate_hz(self.relu(self.lin_gate_zh(z_prev)))
        )
        proposed = self.lin_prop_hz(self.relu(self.lin_prop_zh(z_prev)))
        loc = (1.0 - gate) * self.lin_residual(z_prev) + gate * proposed
        scale = self.softplus(self.lin_scale(self.relu(proposed))) + 1e-5
        return loc, scale


class Combiner(nn.Module):
    """Inference network: fuses z_{t-1} and backward RNN context.

    Produces q(z_t | z_{t-1}, x_{t:T}) parameters by averaging the
    tanh-projected previous latent with the RNN hidden state h_t.
    """

    def __init__(self, z_dim: int, rnn_dim: int) -> None:
        super().__init__()
        self.lin_z_h = nn.Linear(z_dim, rnn_dim)
        self.lin_h_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_h_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(
        self, z_prev: torch.Tensor, h_rnn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = 0.5 * (self.tanh(self.lin_z_h(z_prev)) + h_rnn)
        loc = self.lin_h_loc(h)
        scale = self.softplus(self.lin_h_scale(h)) + 1e-5
        return loc, scale


# ---------------------------------------------------------------------------
# DMM Pyro module
# ---------------------------------------------------------------------------


class DMM(nn.Module):
    """Deep Markov Model with gated transitions and amortised variational inference.

    The generative model factorises as::

        p(x_{1:T}, z_{1:T}) = p(z_1) * prod_t p(z_t|z_{t-1}) * p(x_t|z_t)

    The variational guide uses a backward-RNN inference network::

        q(z_{1:T}|x_{1:T}) = prod_t q(z_t | z_{t-1}, h_rnn_t)

    where h_rnn_t encodes the future context x_t, ..., x_T.
    """

    def __init__(self, input_dim: int, config: DMMConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.emitter = Emitter(input_dim, config.z_dim, config.emission_dim)
        self.trans = GatedTransition(config.z_dim, config.transition_dim)
        self.combiner = Combiner(config.z_dim, config.rnn_dim)
        self.rnn = nn.GRU(input_dim, config.rnn_dim, batch_first=True)
        self.z_0 = nn.Parameter(torch.zeros(config.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(config.z_dim))

    def model(
        self,
        x: torch.Tensor,
        x_reversed: torch.Tensor,
        annealing_factor: float = 1.0,
    ) -> None:
        """Pyro generative model p(x, z)."""
        pyro.module("dmm", self)
        T = x.shape[0]
        z_prev: torch.Tensor = self.z_0
        for t in pyro.markov(range(1, T + 1)):
            z_loc, z_scale = self.trans(z_prev)
            with poutine.scale(scale=annealing_factor):
                z_t = pyro.sample(
                    f"z_{t}",
                    dist.Normal(z_loc, z_scale).to_event(1),
                )
            emission_loc, emission_scale = self.emitter(z_t)
            pyro.sample(
                f"obs_x_{t}",
                dist.Normal(emission_loc, emission_scale).to_event(1),
                obs=x[t - 1],
            )
            z_prev = z_t

    def guide(
        self,
        x: torch.Tensor,
        x_reversed: torch.Tensor,
        annealing_factor: float = 1.0,
    ) -> None:
        """Pyro variational guide q(z | x) with backward-RNN inference."""
        pyro.module("dmm", self)
        T = x.shape[0]
        # Process reversed sequence; re-reverse output to forward time
        rnn_out, _ = self.rnn(x_reversed.unsqueeze(0))
        rnn_out = rnn_out.squeeze(0).flip(0)  # (T, rnn_dim)
        z_prev: torch.Tensor = self.z_q_0
        for t in pyro.markov(range(1, T + 1)):
            z_loc, z_scale = self.combiner(z_prev, rnn_out[t - 1])
            with poutine.scale(scale=annealing_factor):
                z_t = pyro.sample(
                    f"z_{t}",
                    dist.Normal(z_loc, z_scale).to_event(1),
                )
            z_prev = z_t

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a sequence to variational posterior means/stds, shape (T, z_dim)."""
        x_rev = x.flip(0)
        rnn_out, _ = self.rnn(x_rev.unsqueeze(0))
        rnn_out = rnn_out.squeeze(0).flip(0)  # (T, rnn_dim)
        z_prev = self.z_q_0
        locs: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        for i in range(x.shape[0]):
            z_loc, z_scale = self.combiner(z_prev, rnn_out[i])
            locs.append(z_loc)
            scales.append(z_scale)
            z_prev = z_loc  # deterministic mean propagation
        return torch.stack(locs), torch.stack(scales)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_dmm(
    returns: pd.DataFrame,
    config: DMMConfig | None = None,
) -> DMMResult:
    """Fit a Deep Markov Model to a panel of asset returns.

    Uses Pyro SVI with ClippedAdam and KL annealing.  Input returns are
    standardised to zero mean / unit variance per asset before fitting.

    Parameters
    ----------
    returns : pd.DataFrame
        Dates × assets matrix of linear returns.  NaN rows are dropped.
    config : DMMConfig or None
        Model hyper-parameters.  Defaults to ``DMMConfig()``.

    Returns
    -------
    DMMResult
    """
    if config is None:
        config = DMMConfig()

    if config.random_state is not None:
        torch.manual_seed(config.random_state)

    clean = returns.dropna()
    tickers = list(clean.columns)
    input_dim = len(tickers)

    X_np = clean.to_numpy(dtype=np.float64)
    input_mean: npt.NDArray[np.float64] = X_np.mean(axis=0)
    input_std: npt.NDArray[np.float64] = X_np.std(axis=0) + 1e-8
    X_scaled = (X_np - input_mean) / input_std

    x = torch.tensor(X_scaled, dtype=torch.float32)
    x_reversed = x.flip(0)

    pyro.clear_param_store()
    dmm = DMM(input_dim, config)

    adam = ClippedAdam({
        "lr": config.learning_rate,
        "betas": (0.96, 0.999),
        "clip_norm": 10.0,
        "lrd": 0.99996,
    })
    svi = SVI(dmm.model, dmm.guide, adam, loss=Trace_ELBO())

    elbo_history: list[float] = []
    for epoch in range(config.num_epochs):
        if config.annealing_epochs > 0 and epoch < config.annealing_epochs:
            min_af = config.minimum_annealing_factor
            af = min_af + (1.0 - min_af) * (epoch + 1) / config.annealing_epochs
        else:
            af = 1.0
        loss = cast(float, svi.step(x, x_reversed, annealing_factor=af))
        elbo_history.append(-loss)

    dmm.eval()
    with torch.no_grad():
        locs, scales = dmm.encode(x)

    dates = clean.index
    z_cols = list(range(config.z_dim))
    latent_means = pd.DataFrame(locs.numpy(), index=dates, columns=z_cols)
    latent_stds = pd.DataFrame(scales.numpy(), index=dates, columns=z_cols)

    return DMMResult(
        latent_means=latent_means,
        latent_stds=latent_stds,
        elbo_history=elbo_history,
        model=dmm,
        tickers=tickers,
        input_mean=input_mean,
        input_std=input_std,
    )


def blend_moments_dmm(result: DMMResult) -> tuple[pd.Series, pd.DataFrame]:
    """Project the last latent state through the Emitter to produce blended moments.

    Uses the final variational posterior mean z_T as input to the Emitter,
    then un-standardises the output to the original return scale.  Returns
    the same signature as :func:`blend_moments_by_regime` (HMM) for
    drop-in substitution.

    Parameters
    ----------
    result : DMMResult
        Output of :func:`fit_dmm`.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        ``(mu, cov)`` — expected returns and diagonal covariance matrix
        in the original (un-standardised) return scale.
    """
    z_last = torch.tensor(
        result.latent_means.iloc[-1].to_numpy(dtype=np.float32)
    )
    result.model.eval()
    with torch.no_grad():
        emission_loc, emission_scale = result.model.emitter(z_last)

    loc_arr = emission_loc.numpy().astype(np.float64)
    scale_arr = emission_scale.numpy().astype(np.float64)

    mu_arr = loc_arr * result.input_std + result.input_mean
    cov_arr = np.diag((scale_arr * result.input_std) ** 2)

    mu = pd.Series(mu_arr, index=result.tickers)
    cov = pd.DataFrame(cov_arr, index=result.tickers, columns=result.tickers)
    return mu, cov
