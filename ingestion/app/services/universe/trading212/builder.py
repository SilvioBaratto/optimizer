import logging
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.services.universe.trading212.classification import classify_instrument
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.filters.etf_screen import dedup_etfs_by_isin
from app.services.universe.trading212.protocols import (
    LiveSymbolReporter,
    TickerMapper,
    Trading212ApiClient,
    UniverseRepository,
)


@dataclass
class BuildProgress:
    """Snapshot of per-instrument progress for an in-flight universe build."""

    current: int = 0
    total: int = 0
    current_exchange: str = ""
    current_stock: str = ""
    status: str = ""  # "passed", "failed", "skipped"
    reason: str = ""


@dataclass
class BuildResult:
    """Aggregate outcome of a completed universe build."""

    exchanges_saved: int = 0
    instruments_saved: int = 0
    total_processed: int = 0
    filter_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[BuildProgress], None]


@dataclass
class UniverseBuilder:
    """Orchestrates a full Trading 212 universe build.

    Fetches exchange and instrument metadata from the T212 API, maps each
    instrument to a yfinance ticker, classifies it, and persists it via the
    repository. Applies no investability filtering — that is a fund-layer
    concern.
    """

    config: UniverseBuilderConfig
    api_client: Trading212ApiClient
    ticker_mapper: TickerMapper
    repository: UniverseRepository
    max_workers: int = 20
    batch_size: int = 50
    only_exchanges: list[str] | None = None
    progress_callback: ProgressCallback | None = None
    _schedule_to_exchange: dict[int, dict[str, Any]] = field(
        default_factory=dict, init=False
    )
    _instruments_by_schedule: dict[int, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    _errors: list[str] = field(default_factory=list, init=False)
    _live_symbols: frozenset[str] = field(default_factory=frozenset, init=False)
    _incomplete_types: frozenset[str] = field(default=frozenset(), init=False)

    def build(self) -> BuildResult:
        """Run a full universe build from the Trading 212 API.

        Returns:
            Aggregate counts and any per-instrument errors encountered.
        """
        self._errors = []

        exchanges = self.api_client.get_exchanges()
        instruments = self.api_client.get_instruments()
        self._live_symbols = (
            frozenset(self.api_client.get_live_symbols())
            if isinstance(self.api_client, LiveSymbolReporter)
            else frozenset()
        )
        # Read after get_instruments: a source that enumerates lazily only knows
        # which instrument types it failed to list once it has loaded.
        self._incomplete_types = frozenset(
            getattr(self.api_client, "incomplete_instrument_types", ())
        )
        for incomplete_type in sorted(self._incomplete_types):
            self._errors.append(
                f"{incomplete_type} enumeration incomplete: delisting "
                "reconciliation skipped"
            )

        self._build_schedule_mappings(exchanges, instruments)

        # No investability filtering: every classified + mapped instrument is
        # admitted; liquidity/price/history screening is a fund-layer concern.
        exchange_stocks = self._prepare_exchange_stocks(exchanges)
        exchange_etfs = self._prepare_exchange_etfs(exchanges)

        total = sum(len(insts) for _, insts in exchange_stocks) + sum(
            len(insts) for _, insts in exchange_etfs
        )

        # Delisting reconciliation is scoped per instrument_type so the stock
        # and ETF passes over shared exchanges don't falsely mark each other's
        # instruments as delisted.
        exchange_names, instruments_saved, total_processed = self._process_exchanges(
            exchange_stocks, total, instrument_type="STOCK"
        )
        if exchange_etfs:
            etf_exchange_names, inst_e, proc_e = self._process_exchanges(
                exchange_etfs,
                total,
                current_offset=total_processed,
                instrument_type="ETF",
            )
            # Both passes upsert the same exchange rows; count each row once.
            exchange_names |= etf_exchange_names
            instruments_saved += inst_e
            total_processed += proc_e

        return BuildResult(
            exchanges_saved=len(exchange_names),
            instruments_saved=instruments_saved,
            total_processed=total_processed,
            filter_stats={},
            errors=self._errors.copy(),
        )

    def fetch_metadata(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return raw exchange and instrument lists from the T212 API."""
        exchanges = self.api_client.get_exchanges()
        instruments = self.api_client.get_instruments()
        return exchanges, instruments

    def get_exchange_stocks(
        self, exchanges: list[dict[str, Any]], instruments: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Return (exchange, stocks) pairs after building the schedule mappings."""
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

            # only_exchanges set means a targeted debug run; bypass normal filter.
            if self.only_exchanges is not None:
                if exchange_name not in self.only_exchanges:
                    continue
            else:
                if exchange_name not in allowed_exchanges:
                    continue

            all_exchange_instruments = []
            for schedule in ex.get("workingSchedules", []):
                schedule_id = schedule["id"]
                schedule_instruments = self._instruments_by_schedule.get(
                    schedule_id, []
                )
                all_exchange_instruments.extend(schedule_instruments)

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
    ) -> tuple[set[str], int, int]:
        saved_exchange_names: set[str] = set()
        total_instruments_saved = 0
        processed_here = 0

        for ex_data, instruments in exchange_stocks:
            exchange_dto = self.repository.save_exchange(ex_data)
            saved_exchange_names.add(ex_data["name"])

            # Snapshot before the build so absent tickers can be marked
            # delisted after; scoped per instrument_type so the stock and ETF
            # passes don't interfere on shared exchanges.
            tickers_before: set[str] = set()
            if hasattr(self.repository, "get_active_tickers"):
                tickers_before = self.repository.get_active_tickers(
                    exchange_dto.id, instrument_type=instrument_type
                )

            processed = self._process_instruments(
                instruments,
                ex_data["name"],
                total_stocks,
                current_offset + processed_here,
            )
            processed_here += len(instruments)

            if processed:
                saved = self.repository.save_instruments_batch(
                    processed, exchange_dto.id
                )
                total_instruments_saved += saved

            if instrument_type in ex_data.get("incompleteTypes", ()):
                # The source says this venue's list is short, so absence proves
                # nothing; marking the gap delisted would stamp live names.
                self._skip_reconciliation(
                    ex_data["name"], instrument_type, "the source listing is incomplete"
                )
                continue

            # With part of the source missing, "absent" no longer means "delisted".
            if instrument_type in self._incomplete_types:
                continue

            # Every ticker the source returned counts as present, whatever
            # happened to it in mapping: a mapper miss or a per-instrument
            # error is not evidence that the instrument left the venue.
            tickers_present = {i["ticker"] for i in instruments if i.get("ticker")}
            self._mark_delisted_instruments(
                tickers_before,
                tickers_present,
                exchange_dto.id,
                ex_data["name"],
                instrument_type,
            )

        return saved_exchange_names, total_instruments_saved, processed_here

    def _skip_reconciliation(
        self, exchange_name: str, instrument_type: str | None, reason: str
    ) -> None:
        message = (
            f"{exchange_name}: {instrument_type or 'all'} delisting reconciliation "
            f"skipped; {reason}"
        )
        logger.warning(message)
        self._errors.append(message)

    def _mark_delisted_instruments(
        self,
        tickers_before: set[str],
        tickers_seen: set[str],
        exchange_id: Any,
        exchange_name: str,
        instrument_type: str | None,
    ) -> None:
        """Mark instruments that were active but absent from the latest run.

        A ticker the source still reports as trading left the universe through
        dedup, the floor or the venue allowlist, so it is stamped without a
        terminal return; the default -30% is reserved for tickers that vanished.

        Skipped, with a warning in ``BuildResult.errors``, when the absences
        exceed ``config.max_delisting_fraction`` of the venue's active tickers:
        that pattern is a truncated source answer, and marking it would flag
        live instruments as delisted until the next complete build.
        """
        if not hasattr(self.repository, "mark_delisted"):
            return

        dropped = tickers_before - tickers_seen
        if not dropped:
            return

        if (
            len(tickers_before) >= self.config.delisting_guard_min_tickers
            and len(dropped) / len(tickers_before) > self.config.max_delisting_fraction
        ):
            self._skip_reconciliation(
                exchange_name,
                instrument_type,
                f"{len(dropped)} of {len(tickers_before)} active tickers are "
                "missing, which looks like a partial source answer",
            )
            return

        today = date.today()
        for ticker in dropped:
            still_trading: dict[str, Any] = (
                {"delisting_return": None} if ticker in self._live_symbols else {}
            )
            marked = self.repository.mark_delisted(
                ticker=ticker,
                exchange_id=exchange_id,
                delisted_at=today,
                **still_trading,
            )
            if marked:
                logger.info("Marked instrument %s as delisted on %s", ticker, today)

    def _process_instruments(
        self,
        instruments: list[dict[str, Any]],
        exchange_name: str,
        total_stocks: int,
        current_offset: int,
    ) -> list[dict[str, Any]]:
        processed = []
        local_count = 0

        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
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
                    if status == "error":
                        self._record_instrument_error(inst, reason)

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

                except CancelledError:
                    # The progress callback aborts a build whose job row was
                    # reaped; recording that as a per-instrument error would
                    # only trigger a second callback and a spurious error.
                    raise
                except Exception as e:
                    # The worker normally reports failures as an "error" row;
                    # this catches anything that escaped it (or the callback).
                    self._record_instrument_error(inst, str(e))

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
        finally:
            # Leaving the loop early (a reaped job) must drop the queued
            # instruments instead of draining the whole pool on exit.
            executor.shutdown(wait=True, cancel_futures=True)

        return processed

    def _record_instrument_error(self, instrument: dict[str, Any], reason: str) -> None:
        message = f"{instrument.get('shortName', 'unknown')}: {reason}"
        logger.warning("Instrument failed during universe build: %s", message)
        self._errors.append(message)

    def _process_single_instrument(
        self, instrument: dict[str, Any], exchange_name: str
    ) -> tuple[dict[str, Any] | None, str, str]:
        try:
            short_name = instrument.get("shortName", "unknown")

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

            classification = classify_instrument(
                instrument.get("name"), instrument.get("type")
            )
            if classification is None:
                return None, "failed", "Not an investable asset class"
            instrument_data["assetClass"] = classification.asset_class
            instrument_data["fiSubclass"] = classification.fi_subclass
            instrument_data["durationBucket"] = classification.duration_bucket

            yf_ticker = self.ticker_mapper.discover(short_name, exchange_name)

            if not yf_ticker:
                return None, "failed", "No yfinance ticker found"

            instrument_data["yfinanceTicker"] = yf_ticker

            # No investability filtering — a classified, mapped instrument is
            # admitted as-is. Screening (liquidity/price/history) is a downstream
            # fund-layer concern computed on the price panel.
            return instrument_data, "passed", "Admitted (no filtering)"

        except Exception as e:
            # Returned rather than raised so one bad row cannot abort the batch;
            # _process_instruments records it into BuildResult.errors.
            return None, "error", str(e)
