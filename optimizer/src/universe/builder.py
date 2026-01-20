"""
Universe Builder - Main orchestrator for universe construction.

Coordinates all components to build the stock universe:
1. Fetch from Trading212 API
2. Map tickers to yfinance
3. Apply filters
4. Save to database

Follows Dependency Inversion Principle - depends on abstractions (protocols),
not concrete implementations.

Note: This module is intentionally silent (no logging/print).
All user-facing output is handled by the CLI layer.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from optimizer.config.universe_builder_config import UniverseBuilderConfig
from optimizer.domain.protocols.universe import (
    FilterPipeline,
    TickerMapper,
    UniverseRepository,
    Trading212ApiClient,
)
from optimizer.src.universe.services.ticker_mapper import YFinanceTickerMapper


@dataclass
class BuildProgress:
    """Progress information for universe building."""

    current: int = 0
    total: int = 0
    current_exchange: str = ""
    current_stock: str = ""
    status: str = ""  # "passed", "failed", "skipped"
    reason: str = ""


@dataclass
class BuildResult:
    """Result of universe building operation."""

    exchanges_saved: int = 0
    instruments_saved: int = 0
    total_processed: int = 0
    filter_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# Type alias for progress callback
ProgressCallback = Callable[[BuildProgress], None]


@dataclass
class UniverseBuilder:
    """
    Orchestrates universe building with injected dependencies.

    Follows SOLID principles:
    - Single Responsibility: Only orchestration logic
    - Open/Closed: New filters can be added without modification
    - Dependency Inversion: Depends on protocols, not implementations

    Attributes:
        config: UniverseBuilderConfig with all thresholds
        api_client: Trading212ApiClient for fetching metadata
        ticker_mapper: TickerMapper for discovering yfinance tickers
        filter_pipeline: FilterPipeline for applying filters
        repository: UniverseRepository for persistence
        max_workers: Number of concurrent threads for processing
        batch_size: Batch size for database inserts
        skip_filters: If True, skip all filtering
        only_exchanges: If set, only process these exchanges (debug mode)
        progress_callback: Optional callback for progress updates
    """

    config: UniverseBuilderConfig
    api_client: Trading212ApiClient
    ticker_mapper: TickerMapper
    filter_pipeline: FilterPipeline
    repository: UniverseRepository
    max_workers: int = 20
    batch_size: int = 50
    skip_filters: bool = False
    only_exchanges: Optional[List[str]] = None
    progress_callback: Optional[ProgressCallback] = None
    _schedule_to_exchange: Dict[int, Dict[str, Any]] = field(
        default_factory=dict, init=False
    )
    _instruments_by_schedule: Dict[int, List[Dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    _errors: List[str] = field(default_factory=list, init=False)

    def build(self) -> BuildResult:
        """
        Build universe and return detailed result.

        Steps:
        1. Fetch exchanges and instruments from Trading212 API
        2. Filter exchanges by portfolio countries
        3. For each exchange:
           a. Save exchange to database
           b. Process instruments concurrently
           c. Save instruments in batches

        Returns:
            BuildResult with counts and statistics
        """
        self._errors = []

        # Fetch from T212 API
        exchanges = self.api_client.get_exchanges()
        instruments = self.api_client.get_instruments()

        # Build mappings
        self._build_schedule_mappings(exchanges, instruments)

        # Filter exchanges and prepare for processing
        exchange_stocks = self._prepare_exchange_stocks(exchanges)

        # Calculate totals
        total_stocks = sum(len(insts) for _, insts in exchange_stocks)

        # Process exchanges
        exchanges_saved, instruments_saved, total_processed = self._process_exchanges(
            exchange_stocks, total_stocks
        )

        return BuildResult(
            exchanges_saved=exchanges_saved,
            instruments_saved=instruments_saved,
            total_processed=total_processed,
            filter_stats=self.filter_pipeline.get_summary(),
            errors=self._errors.copy(),
        )

    def fetch_metadata(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch exchanges and instruments from API.

        Returns:
            Tuple of (exchanges, instruments) lists
        """
        exchanges = self.api_client.get_exchanges()
        instruments = self.api_client.get_instruments()
        return exchanges, instruments

    def get_exchange_stocks(
        self, exchanges: List[Dict[str, Any]], instruments: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Prepare exchange-stock mappings for processing.

        Returns:
            List of (exchange_data, instruments) tuples
        """
        self._build_schedule_mappings(exchanges, instruments)
        return self._prepare_exchange_stocks(exchanges)

    def _build_schedule_mappings(
        self, exchanges: List[Dict[str, Any]], instruments: List[Dict[str, Any]]
    ) -> None:
        """Build mappings from workingScheduleId to exchange and instruments."""
        self._schedule_to_exchange = {}
        for ex in exchanges:
            for schedule in ex.get("workingSchedules", []):
                self._schedule_to_exchange[schedule["id"]] = ex

        self._instruments_by_schedule = defaultdict(list)
        for inst in instruments:
            schedule_id = inst.get("workingScheduleId")
            if schedule_id:
                self._instruments_by_schedule[schedule_id].append(inst)

    def _prepare_exchange_stocks(
        self, exchanges: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """Filter exchanges and collect their instruments."""
        allowed_exchanges = self.config.get_allowed_exchanges()
        exchange_stocks = []

        for ex in exchanges:
            exchange_name = ex.get("name")
            if not exchange_name:
                continue

            # Debug mode: only process specified exchanges
            if self.only_exchanges is not None:
                if exchange_name not in self.only_exchanges:
                    continue
            else:
                # Normal mode: filter by portfolio countries
                if exchange_name not in allowed_exchanges:
                    continue

            # Collect all instruments for this exchange
            all_exchange_instruments = []
            for schedule in ex.get("workingSchedules", []):
                schedule_id = schedule["id"]
                schedule_instruments = self._instruments_by_schedule.get(schedule_id, [])
                all_exchange_instruments.extend(schedule_instruments)

            # Filter: only STOCK type
            stocks = [i for i in all_exchange_instruments if i.get("type") == "STOCK"]
            if stocks:
                exchange_stocks.append((ex, stocks))

        return exchange_stocks

    def _process_exchanges(
        self,
        exchange_stocks: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
        total_stocks: int,
    ) -> Tuple[int, int, int]:
        """
        Process all exchanges sequentially, instruments concurrently.

        Returns:
            Tuple of (exchanges_saved, instruments_saved, total_processed)
        """
        total_exchanges_saved = 0
        total_instruments_saved = 0
        total_processed = 0

        for ex_data, instruments in exchange_stocks:
            # Save exchange
            exchange_dto = self.repository.save_exchange(ex_data)
            total_exchanges_saved += 1

            # Process instruments concurrently
            processed = self._process_instruments(
                instruments, ex_data["name"], total_stocks, total_processed
            )
            total_processed += len(instruments)

            # Save in batches
            if processed:
                saved = self.repository.save_instruments_batch(
                    processed, exchange_dto.id
                )
                total_instruments_saved += saved

        return total_exchanges_saved, total_instruments_saved, total_processed

    def _process_instruments(
        self,
        instruments: List[Dict[str, Any]],
        exchange_name: str,
        total_stocks: int,
        current_offset: int,
    ) -> List[Dict[str, Any]]:
        """Process instruments concurrently with thread pool."""
        processed = []
        local_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_inst = {
                executor.submit(
                    self._process_single_instrument, inst, exchange_name
                ): inst
                for inst in instruments
            }

            for future in as_completed(future_to_inst):
                inst = future_to_inst[future]
                local_count += 1

                try:
                    result, status, reason = future.result()
                    if result is not None:
                        processed.append(result)

                    # Report progress via callback
                    if self.progress_callback:
                        progress = BuildProgress(
                            current=current_offset + local_count,
                            total=total_stocks,
                            current_exchange=exchange_name,
                            current_stock=inst.get("shortName", "unknown"),
                            status=status,
                            reason=reason,
                        )
                        self.progress_callback(progress)

                except Exception as e:
                    error_msg = f"{inst.get('shortName', 'unknown')}: {e}"
                    self._errors.append(error_msg)

                    if self.progress_callback:
                        progress = BuildProgress(
                            current=current_offset + local_count,
                            total=total_stocks,
                            current_exchange=exchange_name,
                            current_stock=inst.get("shortName", "unknown"),
                            status="error",
                            reason=str(e),
                        )
                        self.progress_callback(progress)

        return processed

    def _process_single_instrument(
        self, instrument: Dict[str, Any], exchange_name: str
    ) -> Tuple[Optional[Dict[str, Any]], str, str]:
        """
        Process a single instrument: discover ticker, fetch data, apply filters.

        Returns:
            Tuple of (instrument_data, status, reason)
            - instrument_data: Dict if passed, None if rejected
            - status: "passed", "failed", "skipped"
            - reason: Human-readable explanation
        """
        try:
            short_name = instrument.get("shortName", "unknown")

            # Build instrument data
            instrument_data = {
                "ticker": instrument.get("ticker"),
                "type": instrument.get("type"),
                "isin": instrument.get("isin"),
                "currencyCode": instrument.get("currencyCode"),
                "name": instrument.get("name"),
                "shortName": short_name,
                "maxOpenQuantity": instrument.get("maxOpenQuantity"),
                "addedOn": instrument.get("addedOn"),
                "exchange": exchange_name,
            }

            # Discover yfinance ticker
            yf_ticker = self.ticker_mapper.discover(short_name, exchange_name)

            if not yf_ticker:
                return None, "failed", "No yfinance ticker found"

            instrument_data["yfinanceTicker"] = yf_ticker

            # Skip filtering if requested
            if self.skip_filters:
                return instrument_data, "skipped", "Filters skipped"

            # Fetch data for filtering
            basic_data = self._fetch_filter_data(yf_ticker)
            if not basic_data:
                return None, "failed", "Failed to fetch yfinance data"

            # Apply filters
            passed, reason = self.filter_pipeline.apply(basic_data, yf_ticker)

            if not passed:
                return None, "failed", reason

            return instrument_data, "passed", reason

        except Exception as e:
            return None, "error", str(e)

    def _fetch_filter_data(self, yf_ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch data needed for filtering from yfinance."""
        if isinstance(self.ticker_mapper, YFinanceTickerMapper):
            return self.ticker_mapper.fetch_basic_data(yf_ticker)

        from optimizer.src.yfinance import YFinanceClient

        client = YFinanceClient.get_instance()
        return client.fetch_info(yf_ticker)

    def get_filter_stats(self) -> Dict[str, Dict[str, int]]:
        """Get filter statistics."""
        return self.filter_pipeline.get_summary()
