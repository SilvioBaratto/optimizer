import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { AnalyticsPaneComponent } from './analytics-pane';
import { BuilderStore } from '../state/builder.store';
import { BuilderResultService } from '../builder-result.service';
import type {
  BuilderBacktest,
  BuilderDriftDiagnostics,
  BuilderResult,
} from '../models/builder-result.model';
import type { DriftDiagnostics } from '../../models/drift.model';

function makeBuilderDiagnostics(
  overrides: Partial<DriftDiagnostics> = {},
): BuilderDriftDiagnostics {
  return {
    requestId: 1,
    diagnostics: {
      reconciliation_ok: true,
      reconciliation_delta_pct: 0,
      unmapped_count: 0,
      fx_missing_count: 0,
      target_not_on_broker_count: 0,
      base_currency: 'EUR',
      sum_eur: 0,
      invested_eur: 0,
      delta_eur: 0,
      tolerance_pct: 0.015,
      stale_price_count: 0,
      entries: [],
      ...overrides,
    },
  };
}

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
      BuilderResultService,
    ],
  });
  const store = TestBed.inject(BuilderStore);
  store.setResultStatus('ok');
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

  describe('diagnostics strip', () => {
    it('when driftDiagnostics is null, strip is not rendered', () => {
      const { host } = setup();
      expect(host.querySelector('app-diagnostics-strip')).toBeNull();
      expect(host.querySelector('[data-region="analytics-footer"]')).toBeNull();
    });

    it('when driftDiagnostics is all-clear (zero counts, reconciliation_ok), strip is not rendered', () => {
      const { fixture, store, host } = setup();
      store.setDriftDiagnostics(makeBuilderDiagnostics());
      fixture.detectChanges();
      expect(host.querySelector('app-diagnostics-strip')).toBeNull();
    });

    it('when driftDiagnostics has a non-zero count, strip is rendered in analytics-footer', () => {
      const { fixture, store, host } = setup();
      store.setDriftDiagnostics(
        makeBuilderDiagnostics({ unmapped_count: 2 }),
      );
      fixture.detectChanges();
      expect(host.querySelector('[data-region="analytics-footer"]')).not.toBeNull();
      expect(host.querySelector('app-diagnostics-strip')).not.toBeNull();
    });

    it('when reconciliation_ok is false, strip is rendered even with zero counts', () => {
      const { fixture, store, host } = setup();
      store.setDriftDiagnostics(
        makeBuilderDiagnostics({
          reconciliation_ok: false,
          reconciliation_delta_pct: 0.03,
          sum_eur: 9700,
          invested_eur: 10000,
          delta_eur: 300,
        }),
      );
      fixture.detectChanges();
      expect(host.querySelector('app-diagnostics-strip')).not.toBeNull();
      expect(host.querySelector('[data-region="diagnostics-reconciliation"]')).not.toBeNull();
    });
  });

  describe('live-state overlay', () => {
    it('when resultStatus is "running", overlay-loading is present inside analytics-pane and no metric cards are in the DOM', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('running');
      fixture.detectChanges();

      const pane = host.querySelector('[data-region="analytics-pane"]');
      expect(pane).not.toBeNull();
      expect(pane!.querySelector('[data-region="overlay-loading"]')).not.toBeNull();
      expect(host.querySelectorAll('[data-card="metric"]').length).toBe(0);
    });

    it('when resultStatus is "error", overlay-error is present and clicking [data-action="retry"] calls store.triggerResultRun exactly once', () => {
      const { fixture, store, host } = setup();
      const triggerSpy = spyOn(store, 'triggerResultRun');

      store.setResultStatus('error');
      fixture.detectChanges();

      expect(host.querySelector('[data-region="overlay-error"]')).not.toBeNull();
      const btn = host.querySelector<HTMLButtonElement>('[data-action="retry"]');
      expect(btn).not.toBeNull();
      btn!.click();

      expect(triggerSpy).toHaveBeenCalledTimes(1);
    });

    it('when resultStatus is "idle", overlay-idle is present and no metric cards are in the DOM', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('idle');
      fixture.detectChanges();

      expect(host.querySelector('[data-region="overlay-idle"]')).not.toBeNull();
      expect(host.querySelectorAll('[data-card="metric"]').length).toBe(0);
    });

    it('when resultStatus is "stale", all six metric cards are present and overlay-stale-badge is present', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('stale');
      fixture.detectChanges();

      expect(host.querySelectorAll('[data-card="metric"]').length).toBe(6);
      expect(host.querySelector('[data-region="overlay-stale-badge"]')).not.toBeNull();
    });

    it('when resultStatus is "ok" with populated result and backtest, all six cards render their formatted values and no overlay region is present', () => {
      const { fixture, store, host } = setup();
      store.setResult(
        makeResult({
          annualized_return: 0.12,
          annualized_volatility: 0.18,
          annualized_sharpe_ratio: 0.85,
          max_drawdown: -0.32,
        }),
      );
      store.setBacktest(makeBacktest({ turnover: 0.15, cost_bps_actual: 8.5 }));
      store.setResultStatus('ok');
      fixture.detectChanges();

      expect(host.querySelectorAll('[data-card="metric"]').length).toBe(6);
      expect(cardByIdValueText(host, 'return')).toBe('12.00%');
      expect(cardByIdValueText(host, 'sharpe')).toBe('0.85');
      expect(host.querySelector('[data-region="overlay-loading"]')).toBeNull();
      expect(host.querySelector('[data-region="overlay-error"]')).toBeNull();
      expect(host.querySelector('[data-region="overlay-idle"]')).toBeNull();
      expect(host.querySelector('[data-region="overlay-stale-badge"]')).toBeNull();
    });
  });
});
