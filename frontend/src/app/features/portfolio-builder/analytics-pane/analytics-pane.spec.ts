import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { AnalyticsPaneComponent } from './analytics-pane';
import { BuilderStore } from '../builder.store';
import type {
  BuilderBacktest,
  BuilderResult,
} from '../builder-result.model';

function makeResult(metrics: Record<string, unknown>): BuilderResult {
  return {
    runId: 'r-1',
    weights: {},
    metrics,
    optimizerType: 'mean_risk',
    createdAt: '2026-01-01T00:00:00Z',
  };
}

function makeBacktest(stats: Record<string, number | null>): BuilderBacktest {
  return {
    runId: 'r-1',
    summaryStats: stats,
    equityCurve: {},
    createdAt: '2026-01-01T00:00:00Z',
  };
}

function setup(): {
  fixture: ComponentFixture<AnalyticsPaneComponent>;
  store: BuilderStore;
  host: HTMLElement;
} {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      BuilderStore,
    ],
  });
  const store = TestBed.inject(BuilderStore);
  const fixture = TestBed.createComponent(AnalyticsPaneComponent);
  fixture.detectChanges();
  return { fixture, store, host: fixture.nativeElement as HTMLElement };
}

function cardByIdValueText(host: HTMLElement, id: string): string {
  const cell = host
    .querySelector<HTMLElement>(`[data-card-id="${id}"] [data-cell="value"]`);
  return cell?.textContent?.trim() ?? '';
}

function cardByIdValueClassList(host: HTMLElement, id: string): DOMTokenList {
  const cell = host
    .querySelector<HTMLElement>(`[data-card-id="${id}"] [data-cell="value"]`);
  if (!cell) throw new Error(`card ${id} value cell missing`);
  return cell.classList;
}

describe('AnalyticsPaneComponent', () => {
  it('when mounted, six metric cards are rendered inside the analytics-pane region', () => {
    const { host } = setup();
    expect(host.querySelector('[data-region="analytics-pane"]')).not.toBeNull();
    expect(host.querySelectorAll('[data-card="metric"]').length).toBe(6);
  });

  it('when result and backtest are null, all six cards show the dash placeholder', () => {
    const { host } = setup();
    for (const id of ['return', 'vol', 'sharpe', 'maxdd', 'turnover', 'cost']) {
      expect(cardByIdValueText(host, id)).toBe('--');
    }
  });

  it('when result has metrics and backtest has stats, all six cards render formatted values', () => {
    const { fixture, store, host } = setup();
    store.setResult(
      makeResult({
        annualized_return: 0.12,
        annualized_volatility: 0.18,
        annualized_sharpe_ratio: 0.85,
        max_drawdown: -0.32,
      }),
    );
    store.setBacktest(
      makeBacktest({ turnover: 0.15, cost_bps_actual: 8.5 }),
    );
    fixture.detectChanges();

    expect(cardByIdValueText(host, 'return')).toBe('12.00%');
    expect(cardByIdValueText(host, 'vol')).toBe('18.00%');
    expect(cardByIdValueText(host, 'sharpe')).toBe('0.85');
    expect(cardByIdValueText(host, 'maxdd')).toBe('-32.00%');
    expect(cardByIdValueText(host, 'turnover')).toBe('15.00%');
    expect(cardByIdValueText(host, 'cost')).toBe('8.50');
  });

  it('when turnover key is mean_turnover, that fallback key is used', () => {
    const { fixture, store, host } = setup();
    store.setBacktest(makeBacktest({ mean_turnover: 0.22 }));
    fixture.detectChanges();
    expect(cardByIdValueText(host, 'turnover')).toBe('22.00%');
  });

  it('when cost key is cost_bps, that fallback key is used', () => {
    const { fixture, store, host } = setup();
    store.setBacktest(makeBacktest({ cost_bps: 12.0 }));
    fixture.detectChanges();
    expect(cardByIdValueText(host, 'cost')).toBe('12.00');
  });

  it('when backtest stats are missing turnover and cost, those two cards show the dash placeholder', () => {
    const { fixture, store, host } = setup();
    store.setBacktest(makeBacktest({}));
    fixture.detectChanges();
    expect(cardByIdValueText(host, 'turnover')).toBe('--');
    expect(cardByIdValueText(host, 'cost')).toBe('--');
  });

  it('when max_drawdown is negative, the maxdd card value cell carries the loss color class', () => {
    const { fixture, store, host } = setup();
    store.setResult(makeResult({ max_drawdown: -0.4 }));
    fixture.detectChanges();
    expect(cardByIdValueClassList(host, 'maxdd')).toContain('text-loss');
  });

  it('when annualised return is positive, the return card value cell carries the gain color class', () => {
    const { fixture, store, host } = setup();
    store.setResult(makeResult({ annualized_return: 0.08 }));
    fixture.detectChanges();
    expect(cardByIdValueClassList(host, 'return')).toContain('text-gain');
  });

  it('when volatility is populated, the vol card value cell uses the neutral text class (no sign color)', () => {
    const { fixture, store, host } = setup();
    store.setResult(makeResult({ annualized_volatility: 0.18 }));
    fixture.detectChanges();
    const classList = cardByIdValueClassList(host, 'vol');
    expect(classList.contains('text-loss')).toBe(false);
    expect(classList.contains('text-gain')).toBe(false);
    expect(classList.contains('text-flat')).toBe(false);
  });

  it('when a metric value is non-finite, the card shows the dash placeholder', () => {
    const { fixture, store, host } = setup();
    store.setResult(
      makeResult({ annualized_return: Number.NaN, annualized_volatility: 'oops' }),
    );
    fixture.detectChanges();
    expect(cardByIdValueText(host, 'return')).toBe('--');
    expect(cardByIdValueText(host, 'vol')).toBe('--');
  });
});
