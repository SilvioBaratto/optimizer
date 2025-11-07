# Quantitative Portfolio Optimization

This research document provides comprehensive documentation of a regime-adaptive quantitative portfolio optimization framework, split into modular chapter files for easy editing and maintenance.

## Structure

```
research/
├── main.md                                    # YAML metadata and introduction
├── chapters/
│   ├── 01_building_universe.md                # Chapter 1: Building the Universe
│   ├── 02_macro_regime.md                     # Chapter 2: Macroeconomic Regime Analysis
│   ├── 03_stock_signals.md                    # Chapter 3: Stock Signal Generation
│   └── (more chapters to be added)
├── build.sh                                   # Build script to compile all chapters
├── quantitative_portfolio_optimization.pdf    # Output PDF (generated)
├── references.bib                             # Bibliography (optional)
└── README.md                                  # This file
```

## Building the Document

### Quick Build

```bash
./build.sh
```

### Manual Build with Pandoc

```bash
export PATH="/Library/TeX/texbin:$PATH"

pandoc main.md \
    chapters/01_building_universe.md \
    chapters/02_macro_regime.md \
    chapters/03_stock_signals.md \
    -o quantitative_portfolio_optimization.pdf \
    --pdf-engine=xelatex \
    --number-sections \
    --toc \
    --toc-depth=3
```

### With Bibliography

If `references.bib` exists, the build script automatically includes it:

```bash
pandoc main.md chapters/*.md \
    -o quantitative_portfolio_optimization.pdf \
    --bibliography=references.bib \
    --citeproc \
    --pdf-engine=xelatex
```

### Build Individual Chapters

To compile a single chapter for quick review:

```bash
pandoc main.md chapters/01_building_universe.md -o chapter1.pdf --pdf-engine=xelatex
```

## Editing Workflow

1. **Edit individual chapter files** in the `chapters/` directory
2. **Run `./build.sh`** to compile the complete PDF
3. **Version control**: Each chapter is independently tracked in git

## Chapter Contents

### Chapter 1: Building the Universe (01_building_universe.md)
- Conceptual framework for universe construction
- Data sources and geographic scope (Trading212, yfinance)
- Ticker mapping methodology
- Institutional filtering framework (market cap, liquidity, price filters)
- Data completeness requirements
- Pipeline architecture and processing strategy
- Quality assurance and validation
- Performance characteristics and scalability
- Challenges and limitations

### Chapter 2: Macroeconomic Regime Analysis (02_macro_regime.md)
- Business cycle framework (Early, Mid, Late, Recession)
- Investment implications of regime classification
- Multi-source economic data integration (Il Sole 24 Ore, FRED, Trading Economics)
- Economic indicators and signal construction (yield curve, PMI, credit spreads, VIX)
- AI-enhanced classification methodology using LLMs
- News-enhanced analysis and sentiment integration
- Regime tracking and transition detection
- Validation and performance assessment
- Challenges and limitations

### Chapter 3: Stock Signal Generation (03_stock_signals.md)
- Multi-factor asset pricing framework (Value, Momentum, Quality, Growth)
- Cross-sectional standardization paradigm
- Factor construction methodology (B/P, E/P, S/P, D/P, ROE, margins, Sharpe ratio)
- Seven-pass cross-sectional architecture (iterative outlier removal)
- Pass-by-pass methodology (raw data → robust statistics → z-score recalculation)
- Winsorization and StandardScaler normalization
- Factor correlation validation and expected patterns
- Percentile-based signal classification (quintile system)
- Momentum filters and adjustments
- James-Stein shrinkage and composite signal construction
- Database integration (SignalDistribution and StockSignal models)
- Statistical rigor and quality assurance

### Future Chapters (To Be Added)
- Chapter 4: Portfolio Optimization
- Chapter 5: Implementation and Results
- Chapter 6: Conclusions and Future Work

## Requirements

- **Pandoc**: Document converter (install via Homebrew: `brew install pandoc`)
- **MacTeX**: LaTeX distribution for PDF generation (install from https://www.tug.org/mactex/)
- **XeLaTeX**: Included with MacTeX

## Tips

- **Edit safely**: Each chapter is independent - breaking one won't affect others
- **Fast iteration**: Compile individual chapters during editing for quick feedback
- **Clean separation**: Content separated from metadata in main.md
- **Version control friendly**: Git diffs show exactly which chapter changed
- **Mathematical notation**: Full LaTeX math support via `$$` blocks
- **Bibliography**: Add citations to `references.bib` and use `[@key]` syntax in markdown

## Adding New Chapters

1. Create new file in `chapters/` with naming convention `XX_chapter_name.md`
2. Add chapter to `build.sh` in the pandoc command
3. Update this README with chapter description
4. Run `./build.sh` to verify compilation

## Notes

- The document was originally a monolithic `project_summary.md` file, now split for modularity
- Each chapter is a standalone markdown file with proper heading hierarchy
- YAML frontmatter and abstract remain in `main.md` only
- The build script combines all chapters in order to produce `quantitative_portfolio_optimization.pdf`
