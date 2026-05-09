import {
  Component,
  DestroyRef,
  ChangeDetectionStrategy,
  computed,
  inject,
  output,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { Subject } from 'rxjs';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { UniverseService } from '../../services/universe.service';
import type {
  Instrument,
  ScreenPreset,
  UniverseScreenResponse,
} from '../../models/universe.model';

const SCREEN_PRESETS: readonly ScreenPreset[] = [
  'developed_markets',
  'broad_universe',
  'small_cap',
  'large_cap',
];

interface ScreenRow extends Record<string, unknown> {
  ticker: string;
  status: string;
  marketCap: string;
  mcapPercentile: string;
  addv12m: string;
  addv3m: string;
  tradingFrequency: string;
  priceFloor: string;
  tradingHistory: string;
  ipoSeasoning: string;
  financialStatements: string;
}

function flag(value: boolean | undefined): string {
  if (value === undefined) return '—';
  return value ? '✓' : '✗';
}

@Component({
  selector: 'app-universe-panel',
  imports: [FormsModule, DataTableComponent, LucideAngularModule],
  templateUrl: './universe-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class UniversePanelComponent {
  private readonly universe = inject(UniverseService);
  private readonly destroyRef = inject(DestroyRef);

  readonly searchTerm = signal('');
  readonly searchResults = signal<Instrument[]>([]);
  readonly selectedPreset = signal<ScreenPreset>('developed_markets');
  readonly screenLoading = signal<boolean>(false);
  readonly screenError = signal<string | null>(null);
  readonly screenResponse = signal<UniverseScreenResponse | null>(null);

  readonly tickerSelected = output<Instrument>();

  readonly presets = SCREEN_PRESETS;

  private readonly searchSubject = new Subject<string>();

  readonly passingCount = computed(
    () => this.screenResponse()?.passingTickers.length ?? 0,
  );
  readonly totalScreened = computed(() => this.screenResponse()?.totalScreened ?? 0);
  readonly hasScreenResult = computed(() => this.screenResponse() !== null);

  readonly screenColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'status', label: 'Status', sortable: true, align: 'right' },
    { key: 'marketCap', label: 'Market cap', align: 'right' },
    { key: 'mcapPercentile', label: 'Mcap pctile', align: 'right' },
    { key: 'addv12m', label: 'ADDV 12m', align: 'right' },
    { key: 'addv3m', label: 'ADDV 3m', align: 'right' },
    { key: 'tradingFrequency', label: 'Trading freq.', align: 'right' },
    { key: 'priceFloor', label: 'Price floor', align: 'right' },
    { key: 'tradingHistory', label: 'Trading hist.', align: 'right' },
    { key: 'ipoSeasoning', label: 'IPO seasoning', align: 'right' },
    { key: 'financialStatements', label: 'Statements', align: 'right' },
  ];

  readonly screenRows = computed<ScreenRow[]>(() => {
    const res = this.screenResponse();
    if (!res) return [];
    const passing = new Set(res.passingTickers);
    const tickers = Object.keys(res.diagnostics).sort((a, b) => {
      const aPass = passing.has(a);
      const bPass = passing.has(b);
      if (aPass !== bPass) return aPass ? -1 : 1;
      return a.localeCompare(b);
    });
    return tickers.map((ticker) =>
      this.buildRow(ticker, res.diagnostics, passing.has(ticker)),
    );
  });

  constructor() {
    this.universe
      .searchTickers(this.searchSubject.asObservable())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((list) => this.searchResults.set(list.items));
  }

  onSearchInput(value: string): void {
    this.searchTerm.set(value);
    this.searchSubject.next(value);
  }

  setPreset(preset: string): void {
    this.selectedPreset.set(preset as ScreenPreset);
  }

  runScreen(): void {
    this.screenLoading.set(true);
    this.screenError.set(null);
    this.universe
      .screen({ preset: this.selectedPreset() })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.screenResponse.set(res);
          this.screenLoading.set(false);
        },
        error: (err: Error) => {
          this.screenError.set(err.message ?? 'Screen failed');
          this.screenLoading.set(false);
        },
      });
  }

  onTickerClick(instrument: Instrument): void {
    this.tickerSelected.emit(instrument);
  }

  onScreenRowClick(row: Record<string, unknown>): void {
    const ticker = row['ticker'] as string | undefined;
    if (!ticker) return;
    this.universe
      .getInstruments({ search: ticker, page: 1, page_size: 1 })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((list) => {
        const match = list.items.find((i) => i.ticker === ticker) ?? list.items[0];
        if (match) {
          this.tickerSelected.emit(match);
        }
      });
  }

  private buildRow(
    ticker: string,
    diagnostics: Record<string, Record<string, boolean>>,
    isPassing: boolean,
  ): ScreenRow {
    const diag = diagnostics[ticker] ?? {};
    return {
      ticker,
      status: isPassing ? 'Pass' : 'Fail',
      marketCap: flag(diag['market_cap']),
      mcapPercentile: flag(diag['mcap_percentile']),
      addv12m: flag(diag['addv_12m']),
      addv3m: flag(diag['addv_3m']),
      tradingFrequency: flag(diag['trading_frequency']),
      priceFloor: flag(diag['price']),
      tradingHistory: flag(diag['trading_history']),
      ipoSeasoning: flag(diag['ipo_seasoning']),
      financialStatements: flag(diag['financial_statements']),
    };
  }
}
