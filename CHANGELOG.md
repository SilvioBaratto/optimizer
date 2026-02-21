# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Structured logging with `NullHandler` pattern across all modules (#36)
- CI Python version matrix testing (3.10, 3.11, 3.12) (#38)
- Test coverage reporting with Codecov integration (#39)
- This CHANGELOG file (#40)
- CI restructure: parallel jobs, concurrency control, format checking (#41)
- Packaging: project URLs, dependency groups, version strategy (#45)
- Community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CITATION.cff (#46)
- GitHub issue and PR templates (#47)
- Dependabot for dependency and GitHub Actions updates (#48)
- Property-based testing with Hypothesis (#49)
- Developer experience: Makefile and examples directory (#50)
- MkDocs documentation site with API reference (#51)
- Automated release workflow (tag to PyPI publish) (#52)
- Pyright compatibility configuration (#53)

## [0.1.0] - 2026-02-21

### Added

- **preprocessing**: Data validation, outlier treatment, sector imputation, regression imputation, delisting adjustment
- **pre_selection**: Pipeline assembly with skfolio selectors (SelectComplete, DropZeroVariance, DropCorrelated, SelectKExtremes, SelectNonDominated, SelectNonExpiring)
- **moments**: Expected return and covariance estimation (empirical, shrinkage, denoised), HMM regime blending, DMM via Pyro, lognormal scaling
- **views**: Black-Litterman, Entropy Pooling, Opinion Pooling integration frameworks
- **optimization**: Mean-Risk, Risk Budgeting, HRP, HERC, NCO, Maximum Diversification, Benchmark Tracking, Equal-Weighted, Inverse-Volatility, Stacking; robust and distributionally robust variants; regime-conditional risk
- **synthetic**: Vine copula models, synthetic data generation, conditional stress testing
- **validation**: Walk-Forward, Combinatorial Purged CV, Multiple Randomized CV
- **scoring**: Performance scoring for model selection
- **tuning**: Grid search and randomized search with temporal cross-validation
- **rebalancing**: Calendar-based, threshold-based, and hybrid rebalancing; turnover and cost computation
- **pipeline**: End-to-end orchestration from prices to validated weights
- **universe**: Investability screening with hysteresis-based entry/exit thresholds
- **factors**: Factor construction, standardization, composite scoring, stock selection, regime tilts, validation, mimicking portfolios, integration with optimization

[Unreleased]: https://github.com/SilvioBaratto/optimizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SilvioBaratto/optimizer/releases/tag/v0.1.0
