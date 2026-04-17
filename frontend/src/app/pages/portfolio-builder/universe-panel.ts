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
  marketCap: string;
  addv3m: string;
  tradingFrequency: string;
  ipoSeasoning: string;
  priceFloor: string;
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
    { key: 'marketCap', label: 'Market cap', align: 'right' },
    { key: 'addv3m', label: 'ADDV 3m', align: 'right' },
    { key: 'tradingFrequency', label: 'Trading freq.', align: 'right' },
    { key: 'ipoSeasoning', label: 'IPO seasoning', align: 'right' },
    { key: 'priceFloor', label: 'Price floor', align: 'right' },
  ];

  readonly screenRows = computed<ScreenRow[]>(() => {
    const res = this.screenResponse();
    if (!res) return [];
    return res.passingTickers.map((ticker) => this.buildRow(ticker, res.diagnostics));
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

  private buildRow(
    ticker: string,
    diagnostics: Record<string, Record<string, boolean>>,
  ): ScreenRow {
    const diag = diagnostics[ticker] ?? {};
    return {
      ticker,
      marketCap: flag(diag['market_cap']),
      addv3m: flag(diag['addv_3m'] ?? diag['addv_12m']),
      tradingFrequency: flag(diag['trading_frequency']),
      ipoSeasoning: flag(diag['ipo_seasoning'] ?? diag['min_ipo_seasoning']),
      priceFloor: flag(diag['price_us'] ?? diag['price_europe']),
    };
  }
}
