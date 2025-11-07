#!/bin/bash

# Build script for Portfolio Construction Guide
# Combines all chapter files into a single PDF

echo "Building Portfolio Construction Guide..."

# Set PATH to include LaTeX
export PATH="/Library/TeX/texbin:$PATH"

# Compile all chapters into single PDF
pandoc main.md \
    chapters/01_portfolio_construction.md \
    chapters/02_macroeconomic_analysis.md \
    chapters/03_diversification_risk_budgeting.md \
    chapters/04_portfolio_optimization.md \
    chapters/05_quantitative_implementation.md \
    chapters/06_synthesis.md \
    -o guide.pdf \
    --pdf-engine=pdflatex \
    --number-sections \
    --toc

if [ $? -eq 0 ]; then
    echo "✓ Successfully generated guide.pdf"
    ls -lh guide.pdf
else
    echo "✗ Build failed"
    exit 1
fi
