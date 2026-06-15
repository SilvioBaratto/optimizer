/**
 * Contract-parity specs for the Rebalancing page (issue #1016).
 *
 * Each test pins one backend contract: HTTP method, URL interpolation, and
 * the absence of raw template tokens (e.g. literal `{name}` in the URL).
 * Tests are driven through the mounted component so the full service-to-HTTP
 * chain is exercised, not just a unit stub.
 *
 * The suite runs in zoneless mode (provideZonelessChangeDetection) so no
 * fakeAsync/tick() is required — effects fire during detectChanges().
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import {
  NO_ERRORS_SCHEMA,
  provideZonelessChangeDetection,
  signal,
  computed,
} from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  provideHttpClientTesting,
  HttpTestingController,
} from '@angular/common/http/testing';

import { RebalancingComponent } from './rebalancing';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

const PORTFOLIO = 't212';
const PORTFOLIO_ID = 'stub-uuid-0002';

function makeCtx(name: string | null = PORTFOLIO, id: string | null = PORTFOLIO_ID) {
  const $name = signal<string | null>(name);
  const $id   = signal<string | null>(id);
  return {
    currentPortfolioId:   $id,
    currentPortfolioName: computed(() => $name()),
    selectedPortfolio:    computed(() => ($name() ? { id: $id() ?? '', name: $name()! } : null)),
    hasPortfolio:         computed(() => $id() !== null),
    dateRange:            signal({ preset: '1Y' as const, start: new Date('2024-01-01'), end: new Date('2025-01-01') }),
    benchmark:            signal('SPY'),
    activeMode:           signal('backtest' as const),
    isLive:               computed(() => false),
    isBacktest:           computed(() => true),
    isPaper:              computed(() => false),
    dateRangeLabel:       computed(() => '1Y'),
    dateRangeDays:        computed(() => 365),
    setPortfolio:         jasmine.createSpy('setPortfolio'),
    setMode:              jasmine.createSpy('setMode'),
    setPreset:            jasmine.createSpy('setPreset'),
    setCustomRange:       jasmine.createSpy('setCustomRange'),
    setBenchmark:         jasmine.createSpy('setBenchmark'),
    reset:                jasmine.createSpy('reset'),
  };
}

describe('Rebalancing — API contract parity (issue #1016)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;
  let http: HttpTestingController;

  beforeEach(async () => {
    const ctx = makeCtx();
    await TestBed.configureTestingModule({
      imports: [RebalancingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: ctx },
      ],
      schemas: [NO_ERRORS_SCHEMA],
    }).compileComponents();

    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(RebalancingComponent);
  });

  afterEach(() => http.verify());

  function flushAll(body: Record<string, unknown> = {}): void {
    http.match(() => true).forEach(r => r.flush(body));
  }

  // ----- GET /portfolio-analytics/{name}/drift -----

  it('when drift is loaded, GET /portfolio-analytics/{name}/drift is called with interpolated name', () => {
    fixture.detectChanges();

    const req = http.expectOne(r =>
      r.url.includes(`/portfolio-analytics/${PORTFOLIO}/drift`),
    );
    expect(req.request.method)
      .withContext('drift must use GET')
      .toBe('GET');
    expect(req.request.url)
      .withContext('URL must not contain raw {name} token')
      .not.toContain('{name}');
    expect(req.request.url).not.toContain('{portfolio_name}');

    req.flush({ entries: [], breachedCount: 0, threshold: 0.05 });
    flushAll();
  });

  // ----- GET /portfolio/{name}/rebalance-policy -----

  it('when policy list is loaded, GET /portfolio/{name}/rebalance-policy is called with interpolated name', () => {
    fixture.detectChanges();

    const req = http.expectOne(
      r =>
        r.url.includes(`/portfolio/${PORTFOLIO}/rebalance-policy`) &&
        r.method === 'GET',
    );
    expect(req.request.url).not.toContain('{name}');
    expect(req.request.url).not.toContain('{portfolio_name}');

    req.flush({ items: [] });
    flushAll();
  });

  // ----- GET /rebalance/preview/{portfolio_name} -----

  it('when trade preview is loaded, GET /rebalance/preview/{portfolio_name} is called with interpolated name', () => {
    fixture.detectChanges();

    const req = http.expectOne(
      r =>
        r.url.includes(`/rebalance/preview/${PORTFOLIO}`) &&
        r.method === 'GET',
    );
    expect(req.request.url).not.toContain('{portfolio_name}');
    expect(req.request.url).not.toContain('{name}');

    req.flush({ trades: [], estimatedCost: 0, totalTurnover: 0 });
    flushAll();
  });

  // ----- GET /portfolio/{name}/snapshots -----

  it('when snapshot history is loaded, GET /portfolio/{name}/snapshots is called with interpolated name', () => {
    fixture.detectChanges();

    const req = http.expectOne(
      r =>
        r.url.includes(`/portfolio/${PORTFOLIO}/snapshots`) &&
        r.method === 'GET',
    );
    expect(req.request.url).not.toContain('{name}');
    expect(req.request.url).not.toContain('{portfolio_name}');

    req.flush({ items: [] });
    flushAll();
  });

  // ----- universal: no raw tokens in any init request -----

  it('when component initialises, no HTTP request URL contains an uninterpolated template token', () => {
    fixture.detectChanges();

    const all = http.match(() => true);
    all.forEach(r => {
      expect(r.request.url)
        .withContext(`URL "${r.request.url}" must not contain raw {token}`)
        .not.toMatch(/\{[a-z_]+\}/);
      r.flush({});
    });
  });

  // ----- no GET /portfolios (portfolio listing not called) -----

  it('when component initialises, no portfolio listing request is made', () => {
    fixture.detectChanges();

    const all = http.match(() => true);
    const listingReqs = all.filter(
      r => r.request.method === 'GET' && /\/portfolios(\?|$)/.test(r.request.url),
    );
    expect(listingReqs.length)
      .withContext('RebalancingComponent must not call the portfolio listing endpoint')
      .toBe(0);

    all.forEach(r => r.flush({}));
  });
});
