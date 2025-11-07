#!/bin/bash

# Build script for Quantitative Portfolio Optimization
# Combines all chapter files into a single PDF

echo "Building Quantitative Portfolio Optimization..."

# Set PATH to include LaTeX
export PATH="/Library/TeX/texbin:$PATH"

# Check if bibliography file exists
BIB_FILE="references.bib"
BIB_ARGS=""
if [ -f "$BIB_FILE" ]; then
    echo "Found bibliography file: $BIB_FILE"
    BIB_ARGS="--bibliography=$BIB_FILE --citeproc"
fi

# Compile all chapters into single PDF
pandoc main.md \
    chapters/01_building_universe.md \
    chapters/02_macro_regime.md \
    chapters/03_stock_signals.md \
    -o quantitative_portfolio_optimization.pdf \
    --pdf-engine=xelatex \
    --number-sections \
    --toc \
    --toc-depth=3 \
    $BIB_ARGS

if [ $? -eq 0 ]; then
    echo "✓ Successfully generated quantitative_portfolio_optimization.pdf"
    ls -lh quantitative_portfolio_optimization.pdf
else
    echo "✗ Build failed"
    exit 1
fi
