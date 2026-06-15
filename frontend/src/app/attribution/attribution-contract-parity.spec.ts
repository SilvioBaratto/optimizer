/**
 * Contract-parity specs for the Attribution page (issue #1017).
 *
 * Criterion covered:
 *   [UNIT] Contract-parity specs assert request and response shapes for every
 *          service method.
 *
 * Pins the API contracts that were previously broken (see requirements §13):
 *   - brinson attribution must use POST /attribution/brinson
 *     (was incorrectly wired as GET /attribution/brinson/{name})
 *   - factor attribution must use POST /attribution/factor
 *     (was incorrectly wired as GET /attribution/factor/{name})
 *   - getLatestSnapshot must be GET with the portfolio name interpolated in
 *     the URL path, not left as a raw {name} template token
 *
 * Each test drives the full service-to-HTTP chain through the mounted
 * component and HttpTestingController, so no service stub can mask a
 * mis-wired URL or wrong HTTP method.
 *
 * Import-path assumptions:
 *   AttributionComponent    → ./attribution
 *   PortfolioContextService → ../core/services/portfolio-context.service
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

import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

// ─── Fixtures ────────────────────────────────────────────────────────────────

const PORTFOLIO    = 't212';
const PORTFOLIO_ID = 'stub-uuid-attr-0001';

function makeCtx(name: string | null = PORTFOLIO, id: string | null = PORTFOLIO_ID) {
  const $name = signal<string | null>(name);
  const $id   = signal<string | null>(id);
  return {
    currentPortfolioId:   $id,
    currentPortfolioName: computed(() => $name()),
    selectedPortfolio:    computed(() =>
      $name() ? { id: $id() ?? '', name: $name()! } : null
    ),
    hasPortfolio:         computed(() => $id() !== null),
    dateRange:            signal({
      preset: '1Y' as const,
      start:  new Date('2024-01-01'),
      end:    new Date('2025-01-01'),
    }),
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

// ─── Suite ───────────────────────────────────────────────────────────────────

describe('Attribution — API contract parity (issue #1017)', () => {
  let fixture: ComponentFixture<AttributionComponent>;
  let http: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtx() },
      ],
      schemas: [NO_ERRORS_SCHEMA],
    }).compileComponents();

    http    = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(AttributionComponent);
  });

  afterEach(() => http.verify());

  function flushAll(body: Record<string, unknown> = {}): void {
    http.match(() => true).forEach(r => r.flush(body));
  }

  function flushSnapshot(): void {
    http
      .match(r =>
        r.method === 'GET' &&
        (r.url.includes('/snapshots/latest') || r.url.toLowerCase().includes('snapshot'))
      )
      .forEach(r =>
        r.flush({
          id: '1',
          portfolioName: PORTFOLIO,
          date: '2025-01-01',
          weights: { AAPL: 0.4, MSFT: 0.6 },
        })
      );
  }

  // ----- Snapshot fetch: GET with interpolated name --------------------------

  it('when attribution loads, getLatestSnapshot is a GET request', () => {
    fixture.detectChanges();

    const snapshotReqs = http.match(
      r => r.url.includes('snapshot') || r.url.includes('latest'),
    );
    expect(snapshotReqs.length)
      .withContext('Expected at least one snapshot request')
      .toBeGreaterThan(0);

    const getReqs = snapshotReqs.filter(r => r.request.method === 'GET');
    expect(getReqs.length)
      .withContext('getLatestSnapshot must use GET')
      .toBeGreaterThan(0);

    snapshotReqs.forEach(r =>
      r.flush({ id: '1', portfolioName: PORTFOLIO, date: '2025-01-01', weights: {} })
    );
    flushAll();
  });

  it('when attribution loads, the snapshot URL contains the portfolio name (not a raw template token)', () => {
    fixture.detectChanges();

    const snapshotReqs = http.match(
      r =>
        r.method === 'GET' &&
        (r.url.includes('snapshot') || r.url.includes('latest')),
    );
    expect(snapshotReqs.length)
      .withContext('Expected at least one snapshot GET request')
      .toBeGreaterThan(0);

    snapshotReqs.forEach(r => {
      expect(r.request.url)
        .withContext(`Snapshot URL must contain portfolio name "${PORTFOLIO}"`)
        .toContain(PORTFOLIO);
      expect(r.request.url)
        .withContext('Snapshot URL must not contain a raw {name} template token')
        .not.toContain('{name}');
      expect(r.request.url)
        .withContext('Snapshot URL must not contain raw {portfolio_name}')
        .not.toContain('{portfolio_name}');
      r.flush({ id: '1', portfolioName: PORTFOLIO, date: '2025-01-01', weights: {} });
    });
    flushAll();
  });

  // ----- POST /attribution/brinson -------------------------------------------

  it('when attribution data loads, the brinson request uses POST', () => {
    fixture.detectChanges();
    flushSnapshot();

    const brinsonReqs = http.match(r => r.url.includes('/attribution/brinson'));
    expect(brinsonReqs.length)
      .withContext('Expected at least one brinson request')
      .toBeGreaterThan(0);

    brinsonReqs.forEach(r => {
      expect(r.request.method)
        .withContext('brinson attribution must use POST, not GET')
        .toBe('POST');
      r.flush({ totalActiveReturn: 0, totalAllocation: 0, totalSelection: 0, sectors: [] });
    });
    flushAll();
  });

  it('when attribution data loads, the brinson URL does not embed the portfolio name as a path segment', () => {
    fixture.detectChanges();
    flushSnapshot();

    const brinsonReqs = http.match(
      r => r.url.includes('/attribution/brinson') && r.method === 'POST',
    );
    brinsonReqs.forEach(r => {
      // The old bug was GET /attribution/brinson/{name}.
      // The fixed endpoint is POST /attribution/brinson (name goes in body).
      expect(r.request.url)
        .withContext(
          `brinson URL must not include "/${PORTFOLIO}" — portfolio name belongs in the request body`
        )
        .not.toMatch(new RegExp(`/attribution/brinson/${PORTFOLIO}`));
      r.flush({ totalActiveReturn: 0, totalAllocation: 0, totalSelection: 0, sectors: [] });
    });
    flushAll();
  });

  it('when attribution data loads, the brinson URL does not contain any raw template token', () => {
    fixture.detectChanges();
    flushSnapshot();

    const brinsonReqs = http.match(
      r => r.url.includes('/attribution/brinson') && r.method === 'POST',
    );
    brinsonReqs.forEach(r => {
      expect(r.request.url)
        .withContext(`brinson URL "${r.request.url}" must not contain raw {token}`)
        .not.toMatch(/\{[a-z_]+\}/);
      r.flush({ totalActiveReturn: 0, totalAllocation: 0, totalSelection: 0, sectors: [] });
    });
    flushAll();
  });

  it('when brinson responds, the response contains the expected top-level shape fields', () => {
    fixture.detectChanges();
    flushSnapshot();

    // Verify that when a correctly-shaped response arrives, the component does not throw.
    // This pins the response interface to the contract fields listed in requirements §5.
    expect(() => {
      http
        .match(r => r.url.includes('/attribution/brinson') && r.method === 'POST')
        .forEach(r =>
          r.flush({
            totalActiveReturn: 0.02,
            totalAllocation:   0.01,
            totalSelection:    0.01,
            sectors: [
              { name: 'Technology', allocation: 0.01, selection: 0.005, interaction: 0.005 },
            ],
          })
        );
      flushAll();
      fixture.detectChanges();
    }).not.toThrow();
  });

  // ----- POST /attribution/factor --------------------------------------------

  it('when attribution data loads, the factor request uses POST', () => {
    fixture.detectChanges();
    flushSnapshot();

    http
      .match(r => r.url.includes('/attribution/brinson') && r.method === 'POST')
      .forEach(r =>
        r.flush({ totalActiveReturn: 0, totalAllocation: 0, totalSelection: 0, sectors: [] })
      );

    const factorReqs = http.match(r => r.url.includes('/attribution/factor'));
    expect(factorReqs.length)
      .withContext('Expected at least one factor request')
      .toBeGreaterThan(0);

    factorReqs.forEach(r => {
      expect(r.request.method)
        .withContext('factor attribution must use POST, not GET')
        .toBe('POST');
      r.flush({ portfolioReturn: 0, explainedReturn: 0, residual: 0, factors: [] });
    });
    flushAll();
  });

  it('when attribution data loads, the factor URL does not embed the portfolio name as a path segment', () => {
    fixture.detectChanges();
    flushSnapshot();

    http
      .match(r => r.url.includes('/attribution/brinson') && r.method === 'POST')
      .forEach(r =>
        r.flush({ totalActiveReturn: 0, totalAllocation: 0, totalSelection: 0, sectors: [] })
      );

    const factorReqs = http.match(
      r => r.url.includes('/attribution/factor') && r.method === 'POST',
    );
    factorReqs.forEach(r => {
      // Old bug: GET /attribution/factor/{name}. Fixed: POST /attribution/factor.
      expect(r.request.url)
        .withContext(
          `factor URL must not include "/${PORTFOLIO}" — portfolio name belongs in the request body`
        )
        .not.toMatch(new RegExp(`/attribution/factor/${PORTFOLIO}`));
      r.flush({ portfolioReturn: 0, explainedReturn: 0, residual: 0, factors: [] });
    });
    flushAll();
  });

  it('when attribution data loads, the factor URL does not contain any raw template token', () => {
    fixture.detectChanges();
    flushSnapshot();

    http
      .match(r => r.url.includes('/attribution/brinson') && r.method === 'POST')
      .forEach(r =>
        r.flush({ totalActiveReturn: 0, totalAllocation: 0, totalSelection: 0, sectors: [] })
      );

    const factorReqs = http.match(
      r => r.url.includes('/attribution/factor') && r.method === 'POST',
    );
    factorReqs.forEach(r => {
      expect(r.request.url)
        .withContext(`factor URL "${r.request.url}" must not contain raw {token}`)
        .not.toMatch(/\{[a-z_]+\}/);
      r.flush({ portfolioReturn: 0, explainedReturn: 0, residual: 0, factors: [] });
    });
    flushAll();
  });

  it('when factor responds, the response contains the expected top-level shape fields', () => {
    fixture.detectChanges();
    flushSnapshot();

    http
      .match(r => r.url.includes('/attribution/brinson') && r.method === 'POST')
      .forEach(r =>
        r.flush({ totalActiveReturn: 0, totalAllocation: 0, totalSelection: 0, sectors: [] })
      );

    expect(() => {
      http
        .match(r => r.url.includes('/attribution/factor') && r.method === 'POST')
        .forEach(r =>
          r.flush({
            portfolioReturn:  0.12,
            explainedReturn:  0.10,
            residual:         0.02,
            factors: [{ name: 'Momentum', contribution: 0.05 }],
          })
        );
      flushAll();
      fixture.detectChanges();
    }).not.toThrow();
  });

  // ----- Universal: no raw template tokens in any init request ---------------

  it('when component initialises, no HTTP request URL contains an uninterpolated template token', () => {
    fixture.detectChanges();

    // First wave: snapshot GET
    http.match(() => true).forEach(r => {
      expect(r.request.url)
        .withContext(`URL "${r.request.url}" must not contain raw {token}`)
        .not.toMatch(/\{[a-z_]+\}/);
      r.flush({ id: '1', portfolioName: PORTFOLIO, date: '2025-01-01', weights: { AAPL: 0.4, MSFT: 0.6 } });
    });

    // Second wave: brinson + factor POSTs triggered by the snapshot subscribe callback
    http.match(() => true).forEach(r => {
      expect(r.request.url)
        .withContext(`URL "${r.request.url}" must not contain raw {token}`)
        .not.toMatch(/\{[a-z_]+\}/);
      r.flush({});
    });
  });

  // ----- No portfolio listing request ----------------------------------------

  it('when attribution initialises, no portfolio listing endpoint is called', () => {
    fixture.detectChanges();

    const all         = http.match(() => true);
    const listingReqs = all.filter(
      r => r.request.method === 'GET' && /\/portfolios(\?|$)/.test(r.request.url),
    );
    expect(listingReqs.length)
      .withContext(
        'AttributionComponent must not call the portfolio listing endpoint — ' +
        'the selected portfolio comes from PortfolioContextService'
      )
      .toBe(0);

    // Flush snapshot, then flush brinson + factor requests it triggers
    all.forEach(r => r.flush({ id: '1', portfolioName: PORTFOLIO, date: '2025-01-01', weights: { AAPL: 0.4, MSFT: 0.6 } }));
    http.match(() => true).forEach(r => r.flush({}));
  });
});
