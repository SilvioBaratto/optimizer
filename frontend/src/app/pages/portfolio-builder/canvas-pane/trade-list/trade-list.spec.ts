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
        flags: [{ code: 'stale_price', reason: 'tick old', reference: 'AAPL' }],
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
        flags: [
          { code: 'fx_missing', reason: 'no fx', reference: 'CHF' },
          { code: 'unmapped', reason: 'no mapping', reference: 'GOOG' },
        ],
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
        flags: [
          { code: 'fx_missing', reason: 'no fx', reference: 'CHF' },
          { code: 'unmapped', reason: 'no mapping', reference: 'GOOG' },
        ],
      },
      {
        ticker: 'AAPL',
        action: 'sell',
        delta_eur: -1000,
        delta_weight: 0.1,
        est_shares: -6,
        est_cost_eur: 1.0,
        flags: [{ code: 'stale_price', reason: 'tick old', reference: 'AAPL' }],
      },
      {
        ticker: 'MSFT',
        action: 'buy',
        delta_eur: 1500,
        delta_weight: -0.1,
        est_shares: 4,
        est_cost_eur: 1.5,
        flags: [],
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
    expect(host.querySelector('[data-region="trade-table"]')).toBeNull();
  });

  it('when trades has zero rows, empty placeholder renders', () => {
    const { fixture, store, host } = setup();
    store.setTrades({ trades: [] });
    fixture.detectChanges();
    expect(host.querySelector('[data-region="trade-empty"]')).not.toBeNull();
  });

  it('when trades is populated, trade-list, trade-table and trade-footer regions render', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    expect(host.querySelector('[data-region="trade-list"]')).not.toBeNull();
    expect(host.querySelector('[data-region="trade-table"]')).not.toBeNull();
    expect(host.querySelector('[data-region="trade-footer"]')).not.toBeNull();
  });

  it('when trades is populated, rows are sorted by |delta_eur| descending', () => {
    const { fixture, store } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const tickers = fixture.componentInstance.rows().map((r) => r.ticker);
    expect(tickers).toEqual(['MSFT', 'AAPL', 'GOOG']);
  });

  it('when action is BUY/SELL/HOLD, the action badge label matches', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const badgeText = (ticker: string): string => {
      const row = host.querySelector(`[data-row-ticker="${ticker}"]`);
      return row!
        .querySelector('[data-cell="action-badge"]')!
        .textContent!.trim();
    };
    expect(badgeText('AAPL')).toBe('SELL');
    expect(badgeText('MSFT')).toBe('BUY');
    expect(badgeText('GOOG')).toBe('HOLD');
  });

  it('when a row has flags, the flags cell renders one app-diagnostic-badge per flag', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();

    const aaplBadges = host.querySelectorAll(
      '[data-row-ticker="AAPL"] [data-cell="flag-badges"] app-diagnostic-badge',
    );
    expect(aaplBadges.length).toBe(1);

    const googBadges = host.querySelectorAll(
      '[data-row-ticker="GOOG"] [data-cell="flag-badges"] app-diagnostic-badge',
    );
    expect(googBadges.length).toBe(2);
  });

  it('when a row has no flags, the flags cell renders zero badges', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const msftBadges = host.querySelectorAll(
      '[data-row-ticker="MSFT"] [data-cell="flag-badges"] app-diagnostic-badge',
    );
    expect(msftBadges.length).toBe(0);
  });

  it('when row flags include UNMAPPED, the row carries opacity-50 disabled-look class', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const goog = host.querySelector('[data-row-ticker="GOOG"]')!;
    expect(goog.getAttribute('data-row-greyed')).toBe('true');
    expect(goog.className).toContain('opacity-50');
  });

  it('when row flags include TARGET_NOT_ON_BROKER, the row carries opacity-50', () => {
    const { fixture, store } = setup();
    store.setTrades({
      trades: [
        {
          ticker: 'GHOST',
          action: 'buy',
          delta_eur: 100,
          delta_weight: 0.01,
          est_shares: 1,
          est_cost_eur: 100,
          flags: [
            { code: 'target_not_on_broker', reason: 'not on broker', reference: 'GHOST' },
          ],
        },
      ],
    });
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const row = (fixture.nativeElement as HTMLElement).querySelector(
      '[data-row-ticker="GHOST"]',
    )!;
    expect(row.getAttribute('data-row-greyed')).toBe('true');
    expect(row.className).toContain('opacity-50');
  });

  it('when row flags only contain STALE_PRICE, the row is NOT greyed', () => {
    const { fixture, store, host } = setup();
    store.setTrades(sampleTrades());
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const aapl = host.querySelector('[data-row-ticker="AAPL"]')!;
    expect(aapl.getAttribute('data-row-greyed')).toBe('false');
    expect(aapl.className).not.toContain('opacity-50');
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
});
