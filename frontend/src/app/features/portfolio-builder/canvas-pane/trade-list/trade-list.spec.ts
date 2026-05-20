import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { TradeListComponent } from './trade-list';
import { BuilderStore } from '../../builder.store';
import type {
  BuilderDrift,
  BuilderTrades,
} from '../../builder-result.model';
import { ICON_PROVIDER } from '../../../../icons';

function sampleDrift(): BuilderDrift {
  return {
    portfolioName: 'core',
    totals: {
      deployable_eur: 10000,
      total_holdings_eur: 10000,
      total_drift_abs: 0.25,
      buy_eur: 800,
      sell_eur: 200,
    },
    drift: [
      {
        ticker: 'AAPL',
        current_weight: 0.6,
        target_weight: 0.5,
        delta_weight: 0.1,
        eur_value: 6000,
        flags: ['stale_price'],
      },
      {
        ticker: 'MSFT',
        current_weight: 0.3,
        target_weight: 0.4,
        delta_weight: -0.1,
        eur_value: 3000,
        flags: [],
      },
      {
        ticker: 'GOOG',
        current_weight: 0.1,
        target_weight: 0.1,
        delta_weight: 0,
        eur_value: 1000,
        flags: ['fx_missing', 'unmapped'],
      },
    ],
  };
}

function sampleTrades(): BuilderTrades {
  return {
    trades: [
      {
        ticker: 'GOOG',
        action: 'hold',
        delta_eur: 0,
        delta_weight: 0,
        est_shares: 0,
        est_cost_eur: 0,
      },
      {
        ticker: 'AAPL',
        action: 'sell',
        delta_eur: -1000,
        delta_weight: 0.1,
        est_shares: -6,
        est_cost_eur: 1.0,
      },
      {
        ticker: 'MSFT',
        action: 'buy',
        delta_eur: 1500,
        delta_weight: -0.1,
        est_shares: 4,
        est_cost_eur: 1.5,
      },
    ],
  };
}

function setup(): {
  fixture: ComponentFixture<TradeListComponent>;
  store: BuilderStore;
  host: HTMLElement;
} {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      ICON_PROVIDER,
      BuilderStore,
    ],
  });
  const store = TestBed.inject(BuilderStore);
  const fixture = TestBed.createComponent(TradeListComponent);
  fixture.detectChanges();
  return { fixture, store, host: fixture.nativeElement as HTMLElement };
}

describe('TradeListComponent', () => {
  it('when trades signal is null, empty placeholder renders', () => {
    const { host } = setup();
    expect(host.querySelector('[data-region="trade-empty"]')).not.toBeNull();
    expect(host.querySelector('[data-region="trade-list"]')).toBeNull();
    expect(host.querySelector('app-data-table')).toBeNull();
  });

  it('when trades has zero rows, empty placeholder renders', () => {
    const { fixture, store, host } = setup();
    store.setTrades({ trades: [] });
    fixture.detectChanges();
    expect(host.querySelector('[data-region="trade-empty"]')).not.toBeNull();
  });

  it('when trades is populated, trade-list and trade-footer regions render', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    expect(host.querySelector('[data-region="trade-list"]')).not.toBeNull();
    expect(host.querySelector('[data-region="trade-footer"]')).not.toBeNull();
    expect(host.querySelector('app-data-table')).not.toBeNull();
  });

  it('when trades is populated, rows are sorted by |delta_eur| descending', () => {
    const { fixture, store } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const tickers = fixture.componentInstance.rows().map((r) => r.ticker);
    expect(tickers).toEqual(['MSFT', 'AAPL', 'GOOG']);
  });

  it('when trades is populated, action column uses badge map with BUY/SELL/HOLD entries', () => {
    const { fixture, store } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const actionCol = (
      fixture.componentInstance as unknown as {
        columns: ReadonlyArray<{ key: string; badgeMap?: Record<string, { value: string; colorClass: string }> }>;
      }
    ).columns.find((c) => c.key === 'action');
    expect(actionCol).toBeDefined();
    expect(actionCol!.badgeMap?.['buy'].value).toBe('BUY');
    expect(actionCol!.badgeMap?.['sell'].value).toBe('SELL');
    expect(actionCol!.badgeMap?.['hold'].value).toBe('HOLD');
    expect(actionCol!.badgeMap?.['buy'].colorClass).toContain('text-gain');
    expect(actionCol!.badgeMap?.['sell'].colorClass).toContain('text-loss');
    expect(actionCol!.badgeMap?.['hold'].colorClass).toContain('text-text-secondary');
  });

  it('when trades and drift are populated, flags column carries raw enum names joined from DriftRow.flags', () => {
    const { fixture, store } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const byTicker = new Map(
      fixture.componentInstance.rows().map((r) => [r.ticker, r.flags] as const),
    );
    expect(byTicker.get('AAPL')).toBe('stale_price');
    expect(byTicker.get('MSFT')).toBe('');
    expect(byTicker.get('GOOG')).toBe('fx_missing, unmapped');
  });

  it('when drift totals are present, footer exposes turnover %, est. cost, residual cash', () => {
    const { fixture, store } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const footer = fixture.componentInstance.footer();
    expect(footer.turnoverPct).toBe(0.25);
    expect(footer.estCostEur).toBe(1000);
    expect(footer.residualCashEur).toBe(9400);
  });

  it('when drift is null, footer falls back to zeros', () => {
    const { fixture, store } = setup();
    store.setTrades(sampleTrades());
    fixture.detectChanges();
    const footer = fixture.componentInstance.footer();
    expect(footer).toEqual({
      turnoverPct: 0,
      estCostEur: 0,
      residualCashEur: 0,
    });
  });

  it('when trade-footer renders, footer cells display formatted turnover %, est. cost, residual cash', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const turnover = host.querySelector('[data-cell="footer-turnover"]')!;
    const cost = host.querySelector('[data-cell="footer-cost"]')!;
    const residual = host.querySelector('[data-cell="footer-residual"]')!;
    expect(turnover.textContent).toContain('25.00%');
    expect(cost.textContent).toContain('1,000');
    expect(residual.textContent).toContain('9,400');
  });

  it('when data-table is rendered, CSV export button is visible (showExport=true)', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const exportBtn = Array.from(host.querySelectorAll('button')).find((b) =>
      (b.textContent ?? '').toLowerCase().includes('export'),
    );
    expect(exportBtn).toBeDefined();
  });
});
