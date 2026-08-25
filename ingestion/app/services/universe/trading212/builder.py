from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.services.universe.trading212.classification import classify_instrument
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.filters.etf_screen import dedup_etfs_by_isin
from app.services.universe.trading212.protocols import (
    FilterPipeline,
    TickerMapper,
    Trading212ApiClient,
    UniverseRepository,
)
from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper


@dataclass
class BuildProgress:
    current: int = 0
    total: int = 0
    current_exchange: str = ""
    current_stock: str = ""
    status: str = ""  # "passed", "failed", "skipped"
    reason: str = ""


@dataclass
class BuildResult:
    exchanges_saved: int = 0
    instruments_saved: int = 0
    total_processed: int = 0
    filter_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# Type alias for progress callback
ProgressCallback = Callable[[BuildProgress], None]


@dataclass
class UniverseBuilder:
    config: UniverseBuilderConfig
    api_client: Trading212ApiClient
    ticker_mapper: TickerMapper
    filter_pipeline: FilterPipeline
    repository: UniverseRepository
    etf_filter_pipeline: FilterPipeline | None = None
    max_workers: int = 20
    batch_size: int = 50
    skip_filters: bool = False
    only_exchanges: list[str] | None = None
    progress_callback: ProgressCallback | None = None
    _schedule_to_exchange: dict[int, dict[str, Any]] = field(
        default_factory=dict, init=False
    )
    _instruments_by_schedule: dict[int, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    _errors: list[str] = field(default_factory=list, init=False)

    def build(self) -> BuildResult:
        self._errors = []

        # Fetch from T212 API
        exchanges = self.api_client.get_exchanges()
        instruments = self.api_client.get_instruments()

        # Build mappings
        self._build_schedule_mappings(exchanges, instruments)

        # Filter exchanges and prepare for processing (stocks + FI/MA ETFs)
        exchange_stocks = self._prepare_exchange_stocks(exchanges)
        exchange_etfs = (
            self._prepare_exchange_etfs(exchanges)
            if self.etf_filter_pipeline is not None
            else []
        )

        # Calculate totals
        total = sum(len(insts) for _, insts in exchange_stocks) + sum(
            len(insts) for _, insts in exchange_etfs
        )

        # Process stocks, then ETFs (through their own pipeline). Delisting
        # reconciliation is scoped per instrument_type so the two passes over
        # shared exchanges don't mark each other's instruments delisted.
        exchanges_saved, instruments_saved, total_processed = self._process_exchanges(
            exchange_stocks, total, instrument_type="STOCK"
        )
        if exchange_etfs:
            ex_e, inst_e, proc_e = self._process_exchanges(
                exchange_etfs,
                total,
                current_offset=total_processed,
                instrument_type="ETF",
            )
            exchanges_saved += ex_e
            instruments_saved += inst_e
            total_processed += proc_e

        filter_stats = dict(self.filter_pipeline.get_summary())
        if self.etf_filter_pipeline is not None:
            filter_stats.update(self.etf_filter_pipeline.get_summary())

        return BuildResult(
            exchanges_saved=exchanges_saved,
            instruments_saved=instruments_saved,
            total_processed=total_processed,
            filter_stats=filter_stats,
            errors=self._errors.copy(),
        )

    def fetch_metadata(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        exchanges = self.api_client.get_exchanges()
        instruments = self.api_client.get_instruments()
        return exchanges, instruments

    def get_exchange_stocks(
        self, exchanges: list[dict[str, Any]], instruments: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        self._build_schedule_mappings(exchanges, instruments)
        return self._prepare_exchange_stocks(exchanges)

    def _build_schedule_mappings(
        self, exchanges: list[dict[str, Any]], instruments: list[dict[str, Any]]
    ) -> None:
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
        self, exchanges: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
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
                schedule_instruments = self._instruments_by_schedule.get(
                    schedule_id, []
                )
                all_exchange_instruments.extend(schedule_instruments)

            # Filter: only STOCK type
            stocks = [i for i in all_exchange_instruments if i.get("type") == "STOCK"]
            if stocks:
                exchange_stocks.append((ex, stocks))

        return exchange_stocks

    def _prepare_exchange_etfs(
        self, exchanges: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Fixed-income + multi-asset ETFs across the (broader) ETF exchange set,
        classifiable (equity/leveraged ETFs excluded) and deduped to one listing
        per ISIN on the most-preferred exchange."""
        allowed = self.config.get_etf_allowed_exchanges()
        ex_by_name: dict[str, dict[str, Any]] = {}
        candidates: list[tuple[str, dict[str, Any]]] = []

        for ex in exchanges:
            name = ex.get("name")
            if not name:
                continue
            # Honour the debug-mode exchange restriction, else the ETF exchange set.
            if self.only_exchanges is not None:
                if name not in self.only_exchanges:
                    continue
            elif name not in allowed:
                continue
            ex_by_name[name] = ex
            for schedule in ex.get("workingSchedules", []):
                for inst in self._instruments_by_schedule.get(schedule["id"], []):
                    if inst.get("type") != "ETF":
                        continue
                    if classify_instrument(inst.get("name"), "ETF") is None:
                        continue
                    candidates.append((name, inst))

        deduped = dedup_etfs_by_isin(candidates, self.config.etf_exchange_preference)

        by_exchange: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for name, inst in deduped:
            by_exchange[name].append(inst)
        return [(ex_by_name[name], insts) for name, insts in by_exchange.items()]

    def _process_exchanges(
        self,
        exchange_stocks: list[tuple[dict[str, Any], list[dict[str, Any]]]],
        total_stocks: int,
        current_offset: int = 0,
        instrument_type: str | None = None,
    ) -> tuple[int, int, int]:
        total_exchanges_saved = 0
        total_instruments_saved = 0
        processed_here = 0

        for ex_data, instruments in exchange_stocks:
            # Save exchange
            exchange_dto = self.repository.save_exchange(ex_data)
            total_exchanges_saved += 1

            # Snapshot active tickers before processing (for delisting detection).
            # Scoped by instrument_type: the stock and ETF passes reconcile their
            # own kind, so the ETF pass never marks a stock delisted (or v.v.) on
            # an exchange shared by both.
            tickers_before: set[str] = set()
            if hasattr(self.repository, "get_active_tickers"):
                tickers_before = self.repository.get_active_tickers(
                    exchange_dto.id, instrument_type=instrument_type
                )

            # Process instruments concurrently
            processed = self._process_instruments(
                instruments,
                ex_data["name"],
                total_stocks,
                current_offset + processed_here,
            )
            processed_here += len(instruments)

            # Save in batches
            tickers_saved: set[str] = set()
            if processed:
                saved = self.repository.save_instruments_batch(
                    processed, exchange_dto.id
                )
                total_instruments_saved += saved
                tickers_saved = {d.get("ticker", "") for d in processed}

            # Detect instruments that dropped out of the T212 universe
            self._mark_delisted_instruments(
                tickers_before, tickers_saved, exchange_dto.id
            )

        return total_exchanges_saved, total_instruments_saved, processed_here

    def _mark_delisted_instruments(
        self,
        tickers_before: set[str],
        tickers_seen: set[str],
        exchange_id: Any,
    ) -> None:
        """Mark instruments that were active but absent from the latest T212 response."""
        if not hasattr(self.repository, "mark_delisted"):
            return

        dropped = tickers_before - tickers_seen
        if not dropped:
            return

        today = date.today()
        for ticker in dropped:
            marked = self.repository.mark_delisted(
                ticker=ticker,
                exchange_id=exchange_id,
                delisted_at=today,
            )
            if marked:
                import logging as _logging

                _logging.getLogger(__name__).info(
                    "Marked instrument %s as delisted on %s", ticker, today
                )

    def _process_instruments(
        self,
        instruments: list[dict[str, Any]],
        exchange_name: str,
        total_stocks: int,
        current_offset: int,
    ) -> list[dict[str, Any]]:
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
                    # Per-instrument failure in a bulk build is surfaced (not
                    # swallowed): appended to self._errors → BuildResult.errors
                    # and reported via the progress callback below.
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
        self, instrument: dict[str, Any], exchange_name: str
    ) -> tuple[dict[str, Any] | None, str, str]:
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

            # Classify into the asset-class taxonomy (STOCK -> equity; ETFs ->
            # fixed_income/multi_asset, or None to reject equity/leveraged ETFs).
            classification = classify_instrument(
                instrument.get("name"), instrument.get("type")
            )
            if classification is None:
                return None, "failed", "Not an investable asset class"
            instrument_data["assetClass"] = classification.asset_class
            instrument_data["fiSubclass"] = classification.fi_subclass
            instrument_data["durationBucket"] = classification.duration_bucket

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

            # Apply the type-appropriate filter pipeline (ETF vs equity)
            pipeline = self._pipeline_for(instrument.get("type"))
            passed, reason = pipeline.apply(basic_data, yf_ticker)

            if not passed:
                return None, "failed", reason

            return instrument_data, "passed", reason

        except Exception as e:
            # Error is returned as the reason tuple, not swallowed: the caller
            # appends it to self._errors (→ BuildResult.errors) for the row.
            return None, "error", str(e)

    def _pipeline_for(self, instrument_type: str | None) -> FilterPipeline:
        """ETFs run the ETF screen; everything else runs the equity pipeline."""
        if instrument_type == "ETF" and self.etf_filter_pipeline is not None:
            return self.etf_filter_pipeline
        return self.filter_pipeline

    def _fetch_filter_data(self, yf_ticker: str) -> dict[str, Any] | None:
        if isinstance(self.ticker_mapper, YFinanceTickerMapper):
            return self.ticker_mapper.fetch_basic_data(yf_ticker)

        from app.services.market_data.yfinance import YFinanceClient

        client = YFinanceClient.get_instance()
        return client.fetch_info(yf_ticker)

    def get_filter_stats(self) -> dict[str, dict[str, int]]:
        return self.filter_pipeline.get_summary()
