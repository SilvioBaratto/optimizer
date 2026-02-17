#!/bin/bash

# Build script for Portfolio Optimization Theory Guide
# Combines all chapter files into a single PDF

echo "Building Portfolio Optimization Theory Guide..."

# Set PATH to include LaTeX
export PATH="/Library/TeX/texbin:$PATH"

# Compile all chapters into single PDF
pandoc main.md \
    chapters/00_stock_preselection.md \
    chapters/01_data_preparation_universe.md \
    chapters/02_moment_estimation_priors.md \
    chapters/03_view_integration.md \
    chapters/04_risk_diversification_hierarchical.md \
    chapters/05_optimization_robust.md \
    chapters/06_validation_pipeline.md \
    bibliography.md \
    -o theory.pdf \
    --pdf-engine=pdflatex \
    --number-sections \
    --toc

if [ $? -eq 0 ]; then
    echo "Successfully generated theory.pdf"
    ls -lh theory.pdf
else
    echo "Build failed"
    exit 1
fi
