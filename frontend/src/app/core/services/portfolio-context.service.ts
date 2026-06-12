import { Injectable, signal, computed, effect, inject } from '@angular/core';
import { toSignal, toObservable } from '@angular/core/rxjs-interop';
import { catchError, distinctUntilChanged, filter, map, of, switchMap } from 'rxjs';

import { PortfolioApiService } from './portfolio-api.service';
import type { PortfolioDto } from '../models/portfolio-api.model';

export type PortfolioMode = 'live' | 'backtest' | 'paper';
export type DateRangePreset = '1D' | '1W' | '1M' | '3M' | '6M' | 'YTD' | '1Y' | '3Y' | '5Y' | 'Max' | 'Custom';

export interface DateRange {
  preset: DateRangePreset;
  start: Date;
  end: Date;
}

const STORAGE_KEY_PORTFOLIO = 'optimizer.currentPortfolioId';

function readStoredPortfolioId(): string | null {
  if (typeof localStorage === 'undefined') return null;
  try {
    return localStorage.getItem(STORAGE_KEY_PORTFOLIO);
  } catch {
    return null;
  }
}

@Injectable({ providedIn: 'root' })
export class PortfolioContextService {
  private readonly portfolioApi = inject(PortfolioApiService);

  readonly activeMode = signal<PortfolioMode>('backtest');
  readonly currentPortfolioId = signal<string | null>(readStoredPortfolioId());
  readonly dateRange = signal<DateRange>(this.computeDateRange('1Y'));
  readonly benchmark = signal('SPY');

  // Only fires the HTTP request on the first null→non-null transition.
  // Switching between non-null ids does NOT re-fetch — the list is cached
  // in this signal while portfolioApi.list()'s shareReplay stays subscribed.
  private readonly portfolioList = toSignal(
    toObservable(this.currentPortfolioId).pipe(
      map((id) => id !== null),
      distinctUntilChanged(),
      filter((hasId) => hasId),
      switchMap(() => this.portfolioApi.list().pipe(catchError(() => of(null)))),
    ),
    { initialValue: null },
  );

  constructor() {
    // Persist portfolio selection across reloads. Storage errors (quota,
    // private mode) must not crash the app, so they're swallowed.
    effect(() => {
      const id = this.currentPortfolioId();
      if (typeof localStorage === 'undefined') return;
      try {
        if (id) {
          localStorage.setItem(STORAGE_KEY_PORTFOLIO, id);
        } else {
          localStorage.removeItem(STORAGE_KEY_PORTFOLIO);
        }
      } catch {
        /* ignore */
      }
    });
  }

  readonly hasPortfolio = computed(() => this.currentPortfolioId() !== null);

  readonly currentPortfolioName = computed<string | null>(() => {
    const id = this.currentPortfolioId();
    if (!id) return null;
    const list = this.portfolioList();
    return list?.items?.find((p) => p.id === id)?.name ?? null;
  });

  readonly selectedPortfolio = computed<PortfolioDto | null>(() => {
    const id = this.currentPortfolioId();
    if (!id) return null;
    const list = this.portfolioList();
    return list?.items?.find((p) => p.id === id) ?? null;
  });

  readonly isLive = computed(() => this.activeMode() === 'live');
  readonly isBacktest = computed(() => this.activeMode() === 'backtest');
  readonly isPaper = computed(() => this.activeMode() === 'paper');

  readonly dateRangeLabel = computed(() => {
    const range = this.dateRange();
    if (range.preset !== 'Custom') return range.preset;
    const fmt = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
    return `${fmt.format(range.start)} - ${fmt.format(range.end)}`;
  });

  readonly dateRangeDays = computed(() => {
    const range = this.dateRange();
    return Math.round((range.end.getTime() - range.start.getTime()) / (1000 * 60 * 60 * 24));
  });

  setMode(mode: PortfolioMode): void {
    this.activeMode.set(mode);
  }

  setPortfolio(id: string | null): void {
    this.currentPortfolioId.set(id);
  }

  setPreset(preset: DateRangePreset): void {
    this.dateRange.set(this.computeDateRange(preset));
  }

  setCustomRange(start: Date, end: Date): void {
    this.dateRange.set({ preset: 'Custom', start, end });
  }

  setBenchmark(ticker: string): void {
    this.benchmark.set(ticker);
  }

  reset(): void {
    this.activeMode.set('backtest');
    this.currentPortfolioId.set(null);
    this.dateRange.set(this.computeDateRange('1Y'));
    this.benchmark.set('SPY');
  }

  private computeDateRange(preset: DateRangePreset): DateRange {
    const end = new Date();
    const start = new Date();

    switch (preset) {
      case '1D':
        start.setDate(start.getDate() - 1);
        break;
      case '1W':
        start.setDate(start.getDate() - 7);
        break;
      case '1M':
        start.setMonth(start.getMonth() - 1);
        break;
      case '3M':
        start.setMonth(start.getMonth() - 3);
        break;
      case '6M':
        start.setMonth(start.getMonth() - 6);
        break;
      case 'YTD':
        start.setMonth(0, 1);
        break;
      case '1Y':
        start.setFullYear(start.getFullYear() - 1);
        break;
      case '3Y':
        start.setFullYear(start.getFullYear() - 3);
        break;
      case '5Y':
        start.setFullYear(start.getFullYear() - 5);
        break;
      case 'Max':
        start.setFullYear(2000, 0, 1);
        break;
      case 'Custom':
        start.setFullYear(start.getFullYear() - 1);
        break;
    }

    return { preset, start, end };
  }
}
