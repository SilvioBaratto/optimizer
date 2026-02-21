# Factor Research

Complete factor research pipeline from construction to optimization integration.

## Pipeline

```
fundamentals → construction → standardization → scoring →
selection → regime tilts → validation → integration
```

## Key Components

- **Construction** -- 17 factor types across 9 groups
- **Standardization** -- Winsorize, z-score/rank-normal, sector neutralize
- **Composite Scoring** -- Equal-weight, IC-weighted, ICIR-weighted, or ML (ridge/GBT)
- **Selection** -- Fixed-count or quantile with buffer-zone hysteresis
- **Regime Tilts** -- GDP/yield-spread macro regime classification
- **Validation** -- IC, Newey-West t-stats, VIF, Benjamini-Hochberg FDR
- **Integration** -- Factor exposure constraints, Black-Litterman views, net alpha
