"""HRP with mutual-information distance and Ward clustering.

Hierarchical Risk Parity on the bundled SP500 dataset using a custom
distance + clustering composition. Prints ranked weights.

Run:
    python examples/hrp_clustering.py
"""

from __future__ import annotations

import pandas as pd
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

from optimizer.cluster import HierarchicalClusteringConfig, LinkageMethodType
from optimizer.distance import DistanceConfig
from optimizer.optimization import HRPConfig, build_hrp


def main() -> None:
    prices = load_sp500_dataset()
    returns = prices_to_returns(prices)

    cfg = HRPConfig(
        distance_config=DistanceConfig.for_mutual_information(n_bins=10),
        clustering_config=HierarchicalClusteringConfig(
            linkage_method=LinkageMethodType.WARD
        ),
    )
    optimizer = build_hrp(cfg)
    optimizer.fit(returns)

    weights = pd.Series(
        optimizer.weights_,
        index=returns.columns,
        name="weight",
    ).sort_values(ascending=False)

    print("HRP weights (mutual_information + Ward):")
    print(weights.round(4).to_string())
    print(f"\nSum: {weights.sum():.4f}")


if __name__ == "__main__":
    main()
