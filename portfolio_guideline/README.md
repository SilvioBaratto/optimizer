# Institutional Portfolio Construction Guide

A comprehensive guide on institutional equity portfolio construction and optimization, split into modular chapter files for easy editing and maintenance.

## Structure

```
portfolio_guideline/
├── main.md                          # YAML metadata and title page
├── chapters/
│   ├── 01_portfolio_construction.md         # Chapter 1: Portfolio Construction Framework
│   ├── 02_macroeconomic_analysis.md        # Chapter 2: Macroeconomic Analysis
│   ├── 03_diversification_risk_budgeting.md # Chapter 3: Diversification & Risk Budgeting
│   ├── 04_portfolio_optimization.md        # Chapter 4: Portfolio Optimization Algorithms
│   ├── 05_quantitative_implementation.md   # Chapter 5: Quantitative Implementation
│   └── 06_synthesis.md                     # Chapter 6: Synthesis
├── build.sh                         # Build script to compile all chapters
├── guide.pdf                        # Output PDF (generated)
└── README.md                        # This file
```

## Building the Guide

### Quick Build

```bash
./build.sh
```

### Manual Build with Pandoc

```bash
export PATH="/Library/TeX/texbin:$PATH"

pandoc main.md \
    chapters/01_portfolio_construction.md \
    chapters/02_macroeconomic_analysis.md \
    chapters/03_diversification_risk_budgeting.md \
    chapters/04_portfolio_optimization.md \
    chapters/05_quantitative_implementation.md \
    chapters/06_synthesis.md \
    -o guide.pdf \
    --pdf-engine=xelatex
```

### Build Individual Chapters

To compile a single chapter for quick review:

```bash
pandoc main.md chapters/01_portfolio_construction.md -o chapter1.pdf --pdf-engine=xelatex
```

## Editing Workflow

1. **Edit individual chapter files** in the `chapters/` directory
2. **Run `./build.sh`** to compile the complete PDF
3. **Version control**: Each chapter is independently tracked in git

## Chapter Contents

### Chapter 1: Portfolio Construction (01_portfolio_construction.md)
- Modern Portfolio Theory foundations
- Black-Litterman framework
- Multi-factor models (Fama-French, Carhart)
- Risk parity approaches
- Integration of fundamental and quantitative analysis

### Chapter 2: Macroeconomic Analysis (02_macroeconomic_analysis.md)
- Business cycle positioning
- Sector rotation strategies
- Quantitative indicators (ISM PMI, yield curve, credit spreads)
- Geographic allocation (DM vs EM)
- Currency hedging decisions

### Chapter 3: Diversification & Risk Budgeting (03_diversification_risk_budgeting.md)
- GICS sector classification
- Concentration limits
- Risk budgeting frameworks
- Factor risk decomposition
- Stress testing methodologies

### Chapter 4: Portfolio Optimization (04_portfolio_optimization.md)
- Mean-variance optimization
- Risk-based optimization (minimum variance, risk parity, HRP)
- Robust optimization (shrinkage estimators, resampled efficiency)
- Transaction cost modeling
- Liquidity constraints

### Chapter 5: Quantitative Implementation (05_quantitative_implementation.md)
- Data preprocessing and outlier treatment
- Factor construction (Value, Growth, Quality, Momentum)
- Risk model building
- Rebalancing frameworks
- Implementation best practices

### Chapter 6: Synthesis (06_synthesis.md)
- Integrated construction process
- Technology infrastructure
- Governance framework
- Institutional imperative

## Requirements

- **Pandoc**: Document converter (install via Homebrew: `brew install pandoc`)
- **MacTeX**: LaTeX distribution for PDF generation (install from https://www.tug.org/mactex/)
- **XeLaTeX**: Included with MacTeX

## Tips

- **Edit safely**: Each chapter is independent - breaking one won't affect others
- **Fast iteration**: Compile individual chapters during editing for quick feedback
- **Clean separation**: Mathematical content separated from metadata in main.md
- **Version control friendly**: Git diffs show exactly which chapter changed
