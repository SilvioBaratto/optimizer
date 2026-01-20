import asyncio
from datetime import date as date_type
from typing import Optional, List
from tqdm.asyncio import tqdm as async_tqdm

from optimizer.database.database import database_manager
from optimizer.database.models.universe import Instrument
from optimizer.src.stock_analyzer.core import MathematicalSignalCalculator

from .database import (
    get_active_instruments,
    get_processed_instrument_ids,
    check_for_incomplete_run,
)
from .processing import (
    process_batch,
    create_batches,
    initialize_stats,
)
from .statistics import calculate_cross_sectional_stats
from . import cross_sectional


class SignalAnalyzer:
    """Async stock signal analyzer with smart resume and batch processing."""

    def __init__(
        self,
        calculator: Optional[MathematicalSignalCalculator] = None,
        concurrent_batches: int = 10,
        stocks_per_batch: int = 10,
    ):
        """Initialize signal analyzer."""
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
        """Main pipeline: Generate signals using institutional 7-pass approach."""
        if signal_date is None:
            signal_date = date_type.today()

        # Fetch active instruments
        with database_manager.get_session() as session:
            instruments = get_active_instruments(session, max_instruments)
            self.stats["total_instruments"] = len(instruments)

            # Smart resume disabled for cross-sectional (need full universe)
            if not enable_cross_sectional:
                # Check for incomplete runs
                incomplete_run = check_for_incomplete_run(session, len(instruments))

                if incomplete_run:
                    resume_date, already_processed = incomplete_run
                    signal_date = resume_date
                    processed_ids = get_processed_instrument_ids(session, signal_date)

                    # Filter out already processed instruments
                    instruments = [i for i in instruments if i.id not in processed_ids]
                    self.stats["skipped_already_processed"] = already_processed
                    self.stats["resumed_from_previous"] = True

        # If not using cross-sectional, use simple approach
        if not enable_cross_sectional:
            await self._run_signal_generation(instruments, signal_date, update_existing)
            self.calculator.finalize_run(
                universe_description=f"Signal Analysis - {len(instruments)} stocks - {signal_date}"
            )
            return

        await self._run_full_seven_pass_analysis(
            instruments, signal_date, update_existing, skip_pass1
        )

        # Finalize calculator distribution
        self.calculator.finalize_run(
            universe_description=f"Trading212 Universe - {signal_date} - {len(instruments)} stocks"
        )

    async def _run_pass1(
        self, instruments: List[Instrument], signal_date: date_type
    ) -> Optional[dict]:
        """Pass 1: Collect raw fundamentals for cross-sectional statistics."""
        # Create batches
        batches = create_batches(instruments, self.stocks_per_batch)

        # Collect fundamentals concurrently
        fundamentals_list = []
        for batch in async_tqdm(batches, desc="Pass 1: Collecting fundamentals"):
            tasks = [
                self.calculator.fetch_raw_fundamentals(i.yfinance_ticker, signal_date)
                for i in batch
                if i.yfinance_ticker
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception) or result is None:
                    continue
                fundamentals_list.append(result)

        # Calculate cross-sectional statistics
        return calculate_cross_sectional_stats(fundamentals_list, enable_stats=True)

    async def _run_signal_generation(
        self,
        instruments: List[Instrument],
        signal_date: date_type,
        update_existing: bool,
    ) -> None:
        """Generate signals for all instruments."""
        # Create batches
        batches = create_batches(instruments, self.stocks_per_batch)

        # Process batches concurrently with progress bar
        for i in range(0, len(batches), self.concurrent_batches):
            batch_group = batches[i : i + self.concurrent_batches]

            tasks = [
                process_batch(
                    self.calculator,
                    batch,
                    signal_date,
                    update_existing,
                    i + j + 1,
                    self.stats,
                )
                for j, batch in enumerate(batch_group)
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_full_seven_pass_analysis(
        self,
        instruments: List[Instrument],
        signal_date: date_type,
        update_existing: bool,
        skip_pass1: bool,
    ) -> None:
        """Execute full 7-pass institutional cross-sectional standardization."""
        import pickle
        from pathlib import Path
        from src.stock_analyzer.risk_free_rate import prefetch_all_risk_free_rates

        risk_free_rates = prefetch_all_risk_free_rates()

        # Update calculator to use cached rates
        self.calculator.risk_free_rates_cache = risk_free_rates

        # Initialize calculator_with_cs as None (will be set if we load cache or run Pass 1B)
        calculator_with_cs = None

        # Define cache file paths
        cache_dir = Path(__file__).parent / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file_pass1 = cache_dir / f"pass1_raw_data_{signal_date}.pkl"
        cache_file_pass1b = cache_dir / f"pass1b_raw_data_cs_{signal_date}.pkl"

        if skip_pass1 and cache_file_pass1b.exists():
            with open(cache_file_pass1b, "rb") as f:
                cached_data = pickle.load(f)
                raw_data = cached_data["raw_data_cs"]
                calculator_with_cs = cached_data["calculator"]

            if len(raw_data) == 0:
                return

        elif skip_pass1 and cache_file_pass1.exists():
            with open(cache_file_pass1, "rb") as f:
                raw_data = pickle.load(f)

            if len(raw_data) == 0:
                return

        else:
            raw_data = []  # List of dicts: {instrument, technical_metrics, info, country}

            # Process in parallel batches with progress bar
            from tqdm.asyncio import tqdm as async_tqdm

            with async_tqdm(
                total=len(instruments), desc="PASS 1: Fetching data", unit="stock"
            ) as pbar:
                for super_batch_start in range(0, len(instruments), self.total_parallel):
                    super_batch_end = min(super_batch_start + self.total_parallel, len(instruments))
                    super_batch = instruments[super_batch_start:super_batch_end]

                    # Process batch concurrently
                    tasks = []
                    for instrument in super_batch:
                        if not instrument.yfinance_ticker:
                            self.stats["skipped_no_ticker"] += 1
                            continue

                        # Fetch raw fundamentals only (no z-score calculation)
                        task = self.calculator.fetch_raw_fundamentals(
                            yf_ticker=instrument.yfinance_ticker,
                            target_date=signal_date,
                        )
                        tasks.append((instrument, task))

                    # Wait for all tasks to complete
                    results = await asyncio.gather(
                        *[task for _, task in tasks], return_exceptions=True
                    )

                    # Store successful results
                    for (instrument, _), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            self.stats["errors"] += 1
                            continue

                        if result is None:
                            continue

                        # Type narrowing: result is now guaranteed to be a tuple
                        technical_metrics, info, country = result  # type: ignore
                        raw_data.append(
                            {
                                "instrument": instrument,
                                "technical_metrics": technical_metrics,
                                "info": info,
                                "country": country,
                            }
                        )

                    pbar.update(len(super_batch))

            if len(raw_data) == 0:
                return

            # Save raw_data to cache for future debugging
            try:
                with open(cache_file_pass1, "wb") as f:
                    pickle.dump(raw_data, f)
            except Exception:
                pass

        if not (skip_pass1 and cache_file_pass1b.exists()):

            cross_sectional_stats = await cross_sectional.calculate_robust_cross_sectional_stats(
                raw_data
            )

            # Create new calculator with cross-sectional statistics and risk-free rates cache
            calculator_with_cs = MathematicalSignalCalculator(
                cross_sectional_stats=cross_sectional_stats,
                risk_free_rates_cache=risk_free_rates,
            )

            raw_data_cs = await cross_sectional.recalculate_with_cross_sectional_stats(
                raw_data,
                signal_date,
                calculator_with_cs,
                self.total_parallel,
                self.stats,
            )

            # Replace raw_data with cross-sectional version
            raw_data = raw_data_cs

            # Save Pass 1B cache for future debugging (skip all the way to Pass 2)
            try:
                cache_data = {
                    "raw_data_cs": raw_data_cs,
                    "calculator": calculator_with_cs,
                }
                with open(cache_file_pass1b, "wb") as f:
                    pickle.dump(cache_data, f)
            except Exception:
                pass

        calc_to_use = calculator_with_cs if calculator_with_cs is not None else self.calculator

        z_standardized = await cross_sectional.apply_cross_sectional_standardization(
            raw_data, calc_to_use
        )

        await cross_sectional.calculate_robust_factor_stats(raw_data)

        await cross_sectional.analyze_factor_correlations(raw_data)

        await cross_sectional.classify_and_save_signals(
            raw_data,
            z_standardized,
            signal_date,
            update_existing,
            calc_to_use,
            self.stats,
        )
