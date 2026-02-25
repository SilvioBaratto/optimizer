import { Injectable, signal, computed } from '@angular/core';

export type PortfolioMode = 'live' | 'backtest' | 'paper';
export type DateRangePreset = '1D' | '1W' | '1M' | '3M' | '6M' | 'YTD' | '1Y' | '3Y' | '5Y' | 'Max' | 'Custom';

export interface DateRange {
  preset: DateRangePreset;
  start: Date;
  end: Date;
}

@Injectable({ providedIn: 'root' })
export class PortfolioContextService {
  readonly activeMode = signal<PortfolioMode>('backtest');
  readonly currentPortfolioId = signal<string | null>(null);
  readonly dateRange = signal<DateRange>(this.computeDateRange('1Y'));
  readonly benchmark = signal('SPY');

  readonly hasPortfolio = computed(() => this.currentPortfolioId() !== null);
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
