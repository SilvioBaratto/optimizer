"""
Signal Analyzer - Main Orchestrator
====================================

Coordinates the entire signal generation pipeline:
1. Fetch active instruments
2. Check for incomplete runs (smart resume)
3. Generate signals using MathematicalSignalCalculator
4. Process in batches with concurrency
5. Save to database
6. Track statistics
"""

import logging
import asyncio
from datetime import date as date_type
from typing import Optional, List
from tqdm.asyncio import tqdm as async_tqdm

from app.database import database_manager
from app.models.universe import Instrument
from src.stock_analyzer.core import MathematicalSignalCalculator

from .database import (
    get_active_instruments,
    get_processed_instrument_ids,
    check_for_incomplete_run,
)
from .processing import (
    process_batch,
    create_batches,
    initialize_stats,
    print_summary as print_processing_summary,
)
from .statistics import calculate_cross_sectional_stats
from . import cross_sectional

logger = logging.getLogger(__name__)


class SignalAnalyzer:
    """
    Async stock signal analyzer with smart resume and batch processing.

    Features:
    - Smart resume: Automatically continues incomplete runs
    - Progress tracking: Saves signals immediately
    - Error resilience: Handles failures gracefully
    - Batch processing: Concurrent signal generation
    """

    def __init__(
        self,
        calculator: Optional[MathematicalSignalCalculator] = None,
        concurrent_batches: int = 10,
        stocks_per_batch: int = 10
    ):
        """
        Initialize signal analyzer.

        Args:
            calculator: Signal calculator instance (creates default if not provided)
            concurrent_batches: Number of parallel batches
            stocks_per_batch: Stocks per batch
        """
        self.calculator = calculator or MathematicalSignalCalculator()
        self.concurrent_batches = concurrent_batches
        self.stocks_per_batch = stocks_per_batch
        self.total_parallel = concurrent_batches * stocks_per_batch
        self.stats = initialize_stats()

    async def analyze_instruments(
        self,
        signal_date: Optional[date_type] = None,
        max_instruments: Optional[int] = None,
        update_existing: bool = True,
        enable_cross_sectional: bool = False,
        skip_pass1: bool = False,
    ) -> None:
        """
        Main pipeline: Generate signals using institutional 7-pass approach.

        INSTITUTIONAL CROSS-SECTIONAL STANDARDIZATION:
        - Pass 1: Fetch raw fundamentals (optimized - no z-scores yet)
        - Pass 1.5: Calculate TRUE cross-sectional statistics (iterative outlier removal)
        - Pass 1B: Recalculate z-scores with robust cross-sectional stats
        - Pass 2: Cross-sectional standardization (winsorize + StandardScaler)
        - Pass 2.5: Calculate robust statistics for factor z-scores
        - Pass 2.6: Factor correlation analysis (validate Chapter 5 expectations)
        - Pass 3: Classify signals and save (with momentum filters)

        This ensures mean=0.000, std=1.000, and perfect 20/20/20/20/20 bucket distribution.

        Args:
            signal_date: Date to generate signals for (defaults to today)
            max_instruments: Maximum number of instruments (for testing)
            update_existing: If True, update existing signals
            enable_cross_sectional: If True, use 7-pass cross-sectional standardization
            skip_pass1: If True, skip Pass 1 and load cached data (for debugging)
        """
        if signal_date is None:
            signal_date = date_type.today()

        logger.info(f"Starting signal generation for {signal_date}")
        logger.info("="*80)
        logger.info("INSTITUTIONAL CROSS-SECTIONAL STANDARDIZATION")
        logger.info("="*80)

        # Fetch active instruments
        with database_manager.get_session() as session:
            instruments = get_active_instruments(session, max_instruments)
            self.stats['total_instruments'] = len(instruments)

            logger.info(f"Found {len(instruments)} active instruments")

            # Smart resume disabled for cross-sectional (need full universe)
            if enable_cross_sectional:
                logger.info("Note: Cross-sectional approach processes entire universe for proper standardization")
            else:
                # Check for incomplete runs
                incomplete_run = check_for_incomplete_run(session, len(instruments))

                if incomplete_run:
                    resume_date, already_processed = incomplete_run
                    signal_date = resume_date
                    processed_ids = get_processed_instrument_ids(session, signal_date)

                    # Filter out already processed instruments
                    instruments = [i for i in instruments if i.id not in processed_ids]
                    self.stats['skipped_already_processed'] = already_processed
                    self.stats['resumed_from_previous'] = True

                    logger.info(
                        f"RESUMING: {len(instruments)} instruments remaining to process"
                    )

        # If not using cross-sectional, use simple approach
        if not enable_cross_sectional:
            await self._run_signal_generation(instruments, signal_date, update_existing)
            self.calculator.finalize_run(
                universe_description=f"Signal Analysis - {len(instruments)} stocks - {signal_date}"
            )
            self.print_summary()
            return

        # ========================================================================
        # FULL 7-PASS INSTITUTIONAL APPROACH
        # ========================================================================

        await self._run_full_seven_pass_analysis(
            instruments, signal_date, update_existing, skip_pass1
        )

        # Finalize calculator distribution
        self.calculator.finalize_run(
            universe_description=f"Trading212 Universe - {signal_date} - {len(instruments)} stocks"
        )

        # Print summary
        self.print_summary()

    async def _run_pass1(
        self, instruments: List[Instrument], signal_date: date_type
    ) -> Optional[dict]:
        """
        Pass 1: Collect raw fundamentals for cross-sectional statistics.

        Args:
            instruments: List of instruments
            signal_date: Signal date

        Returns:
            Cross-sectional statistics or None
        """
        logger.info(f"Collecting fundamentals for {len(instruments)} instruments...")

        # Create batches
        batches = create_batches(instruments, self.stocks_per_batch)

        # Collect fundamentals concurrently
        fundamentals_list = []
        for batch in async_tqdm(batches, desc="Pass 1: Collecting fundamentals"):
            tasks = [
                self.calculator.fetch_raw_fundamentals(i.yfinance_ticker, signal_date)
                for i in batch if i.yfinance_ticker
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception) or result is None:
                    continue
                fundamentals_list.append(result)

        logger.info(f"Collected {len(fundamentals_list)} valid fundamentals")

        # Calculate cross-sectional statistics
        return calculate_cross_sectional_stats(fundamentals_list, enable_stats=True)

    async def _run_signal_generation(
        self,
        instruments: List[Instrument],
        signal_date: date_type,
        update_existing: bool
    ) -> None:
        """
        Generate signals for all instruments.

        Args:
            instruments: List of instruments
            signal_date: Signal date
            update_existing: Update existing signals
        """
        # Create batches
        batches = create_batches(instruments, self.stocks_per_batch)

        logger.info(
            f"Processing {len(instruments)} instruments in {len(batches)} batches "
            f"({self.stocks_per_batch} stocks/batch)"
        )

        # Process batches concurrently with progress bar
        for i in range(0, len(batches), self.concurrent_batches):
            batch_group = batches[i:i + self.concurrent_batches]

            tasks = [
                process_batch(
                    self.calculator,
                    batch,
                    signal_date,
                    update_existing,
                    i + j + 1,
                    self.stats
                )
                for j, batch in enumerate(batch_group)
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_full_seven_pass_analysis(
        self,
        instruments: List[Instrument],
        signal_date: date_type,
        update_existing: bool,
        skip_pass1: bool
    ) -> None:
        """
        Execute full 7-pass institutional cross-sectional standardization.

        This is the production-grade implementation used by leading quant shops.
        """
        import pickle
        from pathlib import Path
        import numpy as np
        from baml_client.types import SignalType
        from src.stock_analyzer.risk_free_rate import prefetch_all_risk_free_rates

        # ====================================================================
        # PRE-FETCH RISK-FREE RATES (Performance optimization)
        # ====================================================================
        logger.info("")
        logger.info("="*80)
        logger.info("PRE-FETCHING RISK-FREE RATES...")
        logger.info("="*80)

        risk_free_rates = prefetch_all_risk_free_rates()

        # Update calculator to use cached rates
        self.calculator.risk_free_rates_cache = risk_free_rates

        logger.info("")

        # Initialize calculator_with_cs as None (will be set if we load cache or run Pass 1B)
        calculator_with_cs = None

        # Define cache file paths
        cache_dir = Path(__file__).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file_pass1 = cache_dir / f"pass1_raw_data_{signal_date}.pkl"
        cache_file_pass1b = cache_dir / f"pass1b_raw_data_cs_{signal_date}.pkl"

        # ====================================================================
        # Check for Pass 1B cache (skip Pass 1, 1.5, and 1B if exists)
        # ====================================================================
        if skip_pass1 and cache_file_pass1b.exists():
            logger.info("")
            logger.info("="*80)
            logger.info(f"🔍 SUPER DEBUG MODE: Loading cached Pass 1B data (with z-scores) from {cache_file_pass1b}")
            logger.info("="*80)

            with open(cache_file_pass1b, 'rb') as f:
                cached_data = pickle.load(f)
                raw_data = cached_data['raw_data_cs']
                calculator_with_cs = cached_data['calculator']

            if len(raw_data) == 0:
                logger.error("❌ Cached Pass 1B data is empty. Aborting.")
                return

            logger.info(f"✅ Loaded {len(raw_data)} stocks with z-scores from Pass 1B cache")
            logger.info("⚠️  Skipping Pass 1, 1.5, and 1B - jumping to Pass 2")
            logger.info("")

        # ====================================================================
        # PASS 1: Fetch raw fundamentals (NO z-score calculation)
        # ====================================================================
        elif skip_pass1 and cache_file_pass1.exists():
            logger.info("")
            logger.info("="*80)
            logger.info(f"🔍 DEBUG MODE: Loading cached Pass 1 data from {cache_file_pass1}")
            logger.info("="*80)

            with open(cache_file_pass1, 'rb') as f:
                raw_data = pickle.load(f)

            if len(raw_data) == 0:
                logger.error("❌ Cached data is empty. Aborting.")
                return

            logger.info(f"✅ Loaded {len(raw_data)} cached stocks from Pass 1")
            logger.info("⚠️  Skipping Pass 1 - using cached data for debugging")

        else:
            if skip_pass1:
                logger.warning(f"⚠️  Cache files not found")
                logger.warning("Falling back to normal Pass 1 execution")

            logger.info("")
            logger.info("="*80)
            logger.info("PASS 1: Fetching raw fundamentals (optimized - no z-scores yet)...")
            logger.info("="*80)

            raw_data = []  # List of dicts: {instrument, technical_metrics, info, country}

            # Process in parallel batches with progress bar
            from tqdm.asyncio import tqdm as async_tqdm

            with async_tqdm(
                total=len(instruments),
                desc="PASS 1: Fetching data",
                unit="stock"
            ) as pbar:
                for super_batch_start in range(0, len(instruments), self.total_parallel):
                    super_batch_end = min(super_batch_start + self.total_parallel, len(instruments))
                    super_batch = instruments[super_batch_start:super_batch_end]

                    # Process batch concurrently
                    tasks = []
                    for instrument in super_batch:
                        if not instrument.yfinance_ticker:
                            self.stats['skipped_no_ticker'] += 1
                            continue

                        # Fetch raw fundamentals only (no z-score calculation)
                        task = self.calculator.fetch_raw_fundamentals(
                            yf_ticker=instrument.yfinance_ticker,
                            target_date=signal_date
                        )
                        tasks.append((instrument, task))

                    # Wait for all tasks to complete
                    results = await asyncio.gather(
                        *[task for _, task in tasks],
                        return_exceptions=True
                    )

                    # Store successful results
                    for i, ((instrument, _), result) in enumerate(zip(tasks, results)):
                        if isinstance(result, Exception):
                            logger.debug(f"Error fetching fundamentals for {instrument.ticker}: {result}")
                            self.stats['errors'] += 1
                            continue

                        if result is None:
                            continue

                        # Type narrowing: result is now guaranteed to be a tuple
                        technical_metrics, info, country = result  # type: ignore
                        raw_data.append({
                            'instrument': instrument,
                            'technical_metrics': technical_metrics,
                            'info': info,
                            'country': country
                        })

                    pbar.update(len(super_batch))

            if len(raw_data) == 0:
                logger.error("No valid data fetched. Aborting.")
                return

            logger.info(f"✅ PASS 1 Complete: {len(raw_data)} stocks fetched")

            # Save raw_data to cache for future debugging
            try:
                with open(cache_file_pass1, 'wb') as f:
                    pickle.dump(raw_data, f)
                logger.info(f"💾 Cached Pass 1 data to {cache_file_pass1} (use --skip-pass1 to load)")
            except Exception as e:
                logger.warning(f"Failed to cache Pass 1 data: {e}")

        # ====================================================================
        # PASS 1.5 & 1B: Only run if we didn't load from Pass 1B cache
        # ====================================================================
        if not (skip_pass1 and cache_file_pass1b.exists()):
            # ====================================================================
            # PASS 1.5: Calculate TRUE Cross-Sectional Statistics
            # ====================================================================

            logger.info("")
            logger.info("="*80)
            logger.info("PASS 1.5: Calculating TRUE cross-sectional statistics (iterative outlier removal)...")
            logger.info("="*80)

            cross_sectional_stats = await cross_sectional.calculate_robust_cross_sectional_stats(raw_data)

            logger.info(f"✅ PASS 1.5 Complete: Cross-sectional statistics calculated from {len(raw_data)} stocks")

            # ====================================================================
            # PASS 1B: Recalculate z-scores with TRUE cross-sectional statistics
            # ====================================================================

            logger.info("")
            logger.info("="*80)
            logger.info("PASS 1B: Calculating z-scores with TRUE cross-sectional standardization...")
            logger.info("="*80)

            # Create new calculator with cross-sectional statistics and risk-free rates cache
            calculator_with_cs = MathematicalSignalCalculator(
                cross_sectional_stats=cross_sectional_stats,
                risk_free_rates_cache=risk_free_rates
            )

            raw_data_cs = await cross_sectional.recalculate_with_cross_sectional_stats(
                raw_data, signal_date, calculator_with_cs, self.total_parallel, self.stats
            )

            logger.info(f"✅ PASS 1B Complete: {len(raw_data_cs)} z-scores recalculated with cross-sectional stats")

            # Replace raw_data with cross-sectional version
            raw_data = raw_data_cs

            # Save Pass 1B cache for future debugging (skip all the way to Pass 2)
            try:
                cache_data = {
                    'raw_data_cs': raw_data_cs,
                    'calculator': calculator_with_cs
                }
                with open(cache_file_pass1b, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"💾 Cached Pass 1B data (with z-scores) to {cache_file_pass1b}")
                logger.info(f"   Next run: use --skip-pass1 to skip directly to Pass 2!")
            except Exception as e:
                logger.warning(f"Failed to cache Pass 1B data: {e}")

        # ====================================================================
        # PASS 2: Cross-sectional standardization (winsorize + StandardScaler)
        # ====================================================================

        logger.info("")
        logger.info("="*80)
        logger.info("PASS 2: Applying cross-sectional standardization (winsorization + scaling)...")
        logger.info("="*80)

        # Use calculator_with_cs if available (from Pass 1B or cache), otherwise use self.calculator
        calc_to_use = calculator_with_cs if calculator_with_cs is not None else self.calculator

        z_standardized = await cross_sectional.apply_cross_sectional_standardization(
            raw_data, calc_to_use
        )

        logger.info(f"✅ PASS 2 Complete: Distribution standardized (mean=0, std=1)")

        # ====================================================================
        # PASS 2.5: Calculate Robust Statistics for Factor Z-Scores
        # ====================================================================

        logger.info("")
        logger.info("="*80)
        logger.info("PASS 2.5: Calculating robust statistics for factor z-scores...")
        logger.info("="*80)

        robust_stats = await cross_sectional.calculate_robust_factor_stats(raw_data)

        logger.info(f"✅ PASS 2.5 Complete: Robust statistics calculated")

        # ====================================================================
        # PASS 2.6: Factor Correlation Analysis
        # ====================================================================

        logger.info("")
        logger.info("="*80)
        logger.info("PASS 2.6: Analyzing factor correlations (validating Chapter 5)...")
        logger.info("="*80)

        await cross_sectional.analyze_factor_correlations(raw_data)

        logger.info(f"✅ PASS 2.6 Complete: Factor correlation analysis done")

        # ====================================================================
        # PASS 3: Classify signals and save to database
        # ====================================================================

        logger.info("")
        logger.info("="*80)
        logger.info("PASS 3: Classifying signals and saving to database...")
        logger.info("="*80)

        await cross_sectional.classify_and_save_signals(
            raw_data, z_standardized, signal_date, update_existing, calc_to_use, self.stats
        )

        logger.info("✅ PASS 3 Complete: All signals saved")
        logger.info("")
        logger.info("="*80)
        logger.info("7-PASS INSTITUTIONAL ANALYSIS COMPLETE")
        logger.info("="*80)

    def print_summary(self) -> None:
        """Print final summary of signal generation run."""
        print_processing_summary(self.stats)
