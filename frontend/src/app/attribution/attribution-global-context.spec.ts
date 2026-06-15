/**
 * Source-blind unit tests for issue #1017.
 * feat(attribution): migrate to global PortfolioContextService
 *                    and source weights from snapshot.
 *
 * Criteria covered (oracle tier in parentheses):
 *   [UNIT] attribution reads portfolioContextService.selectedPortfolio() /
 *          currentPortfolioName(); there is no writable local selectedPortfolio signal
 *   [UNIT] Portfolio weights are sourced from PortfolioApiService.getLatestSnapshot(name),
 *          not a manual portfolio-weights selection UI
 *   [UNIT] No portfolio <select>/dropdown exists in attribution.html;
 *          selection happens only via the global picker
 *   [UNIT] getLatestSnapshot is invoked with the resolved NAME (not the raw UUID);
 *          switching the portfolio refetches the snapshot with the new name
 *   [UNIT] Every wired page has specs covering:
 *            (a) loads data on init
 *            (b) handles API error
 *            (c) reacts to portfolio context change
 *   [UNIT] Every page/panel shows a visible, non-blank error state when its primary
 *          API call returns an error
 *   [UNIT] No unhandled observable errors reach console.error
 *   [UNIT] No API endpoint or parameter names appear in any user-facing UI copy
 *
 * Criteria skipped (oracle: NOT VERIFIABLE):
 *   – Attribution form hint shows the actual portfolio name (NOT VERIFIABLE)
 *   – All tests pass gate (meta criterion; NOT VERIFIABLE from spec file)
 *   – SOLID, clean code (NOT VERIFIABLE: subjective)
 *
 * Invariant-based property tests (TypeScript equivalent of @given):
 *   – For any portfolio name set in context, getLatestSnapshot always receives that
 *     name (never the UUID) — tested across multiple name values sequentially.
 *
 * Import-path assumptions:
 *   AttributionComponent    → ./attribution
 *   PortfolioContextService → ../core/services/portfolio-context.service
 *   PortfolioApiService     → ../core/services/portfolio-api.service
 *   AttributionService      → ./attribution.service
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import {
  NO_ERRORS_SCHEMA,
  provideZonelessChangeDetection,
  signal,
  computed,
} from '@angular/core';
import { of, throwError } from 'rxjs';

import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AttributionService } from './attribution.service';
import { ICON_PROVIDER } from '../icons';

// ─── Constants ────────────────────────────────────────────────────────────────

const PORTFOLIO_ID   = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
const PORTFOLIO_NAME = 't212';
const SECOND_ID      = 'a1b2c3d4-0000-0000-0000-000000000002';
const SECOND_NAME    = 'ib-flex';

const STUB_SNAPSHOT = {
  id: '1',
  portfolioName: PORTFOLIO_NAME,
  date: '2025-01-01',
  weights: { AAPL: 0.5, MSFT: 0.5 },
};

const STUB_BRINSON = {
  totalActiveReturn: 0.01,
  totalAllocation: 0.005,
  totalSelection: 0.005,
  sectors: [],
};

const STUB_FACTOR = {
  portfolioReturn: 0.1,
  explainedReturn: 0.09,
  residual: 0.01,
  factors: [],
};

// ─── Mock factories ───────────────────────────────────────────────────────────

function makeCtx(name: string | null = PORTFOLIO_NAME, id: string | null = PORTFOLIO_ID) {
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
    _setTo: (n: string | null, i: string | null) => {
      $name.set(n);
      $id.set(i);
    },
  };
}

function makePortfolioApi(opts: { snapshotError?: boolean } = {}) {
  return {
    getLatestSnapshot: jasmine
      .createSpy('getLatestSnapshot')
      .and.returnValue(
        opts.snapshotError
          ? throwError(() => new Error('500 Server Error'))
          : of(STUB_SNAPSHOT),
      ),
    listPortfolios: jasmine.createSpy('listPortfolios').and.returnValue(of({ items: [] })),
  };
}

function makeAttributionSvc(opts: {
  brinsonError?: boolean;
  factorError?: boolean;
} = {}) {
  return {
    brinson: jasmine
      .createSpy('brinson')
      .and.returnValue(
        opts.brinsonError
          ? throwError(() => new Error('500 Server Error'))
          : of(STUB_BRINSON),
      ),
    factor: jasmine
      .createSpy('factor')
      .and.returnValue(
        opts.factorError
          ? throwError(() => new Error('500 Server Error'))
          : of(STUB_FACTOR),
      ),
  };
}

type CtxMock = ReturnType<typeof makeCtx>;
type PortfolioApiMock = ReturnType<typeof makePortfolioApi>;
type AttributionSvcMock = ReturnType<typeof makeAttributionSvc>;

async function boot(
  ctx: CtxMock,
  portfolioApi: PortfolioApiMock,
  attrSvc: AttributionSvcMock,
): Promise<ComponentFixture<AttributionComponent>> {
  await TestBed.configureTestingModule({
    imports: [AttributionComponent],
    schemas: [NO_ERRORS_SCHEMA],
    providers: [
      provideZonelessChangeDetection(),
      ICON_PROVIDER,
      { provide: PortfolioContextService, useValue: ctx },
      { provide: PortfolioApiService,     useValue: portfolioApi },
      { provide: AttributionService,      useValue: attrSvc },
    ],
  }).compileComponents();

  const fixture = TestBed.createComponent(AttributionComponent);
  fixture.detectChanges();
  return fixture;
}

// ─── Suite: no portfolio dropdown in template ─────────────────────────────────

describe('AttributionComponent — no portfolio dropdown in template (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "No portfolio <select>/dropdown exists in attribution.html;
   * selection happens only via the global picker."
   */

  it('when attribution renders, no portfolio-labelled <select> element exists in the DOM', async () => {
    const fixture = await boot(makeCtx(), makePortfolioApi(), makeAttributionSvc());

    const el      = fixture.nativeElement as HTMLElement;
    const selects = Array.from(el.querySelectorAll('select'));
    const portfolioSelect = selects.find((s) => {
      const ariaLabel = (s.getAttribute('aria-label') ?? '').toLowerCase();
      const name      = (s.getAttribute('name') ?? '').toLowerCase();
      const id        = (s.getAttribute('id') ?? '').toLowerCase();
      return (
        ariaLabel.includes('portfolio') ||
        name.includes('portfolio') ||
        id.includes('portfolio')
      );
    });
    expect(portfolioSelect)
      .withContext('Found a portfolio-labelled <select>; must use the global picker instead')
      .toBeUndefined();
  });

  it('when attribution renders, no formControlName containing "portfolio" drives a select', async () => {
    const fixture = await boot(makeCtx(), makePortfolioApi(), makeAttributionSvc());

    const el = fixture.nativeElement as HTMLElement;
    const portfolioPicker = el.querySelector(
      'select[formcontrolname*="portfolio" i], [data-portfolio-picker], #portfolio-select',
    );
    expect(portfolioPicker)
      .withContext('no portfolio-picker formControl on a select must exist')
      .toBeNull();
  });
});

// ─── Suite: selectedPortfolio is read-only computed (no writable local signal) ─

describe('AttributionComponent — selectedPortfolio has no .set method (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "attribution reads portfolioContextService.selectedPortfolio() /
   * currentPortfolioName(); there is no writable local selectedPortfolio signal."
   *
   * A writable signal exposes .set(); a computed() signal does not.
   */

  it('when component is created, selectedPortfolio has no .set method', async () => {
    const fixture = await boot(makeCtx(), makePortfolioApi(), makeAttributionSvc());
    const comp = fixture.componentInstance;

    expect(typeof (comp.selectedPortfolio as unknown as { set?: unknown }).set)
      .withContext(
        'selectedPortfolio must not expose .set — it must be read-only computed from context, not a writable local signal'
      )
      .toBe('undefined');
  });
});

// ─── Suite: (a) loads data on init ───────────────────────────────────────────

describe('AttributionComponent — (a) loads data on init (issue #1017 [UNIT])', () => {
  /**
   * Criteria: "Every wired page has specs covering: (a) loads data on init."
   *           "getLatestSnapshot is invoked with the resolved NAME (not the raw UUID)."
   *           "Portfolio weights are sourced from PortfolioApiService.getLatestSnapshot(name)."
   */

  it('when portfolio is in context, getLatestSnapshot is called on init', async () => {
    const portfolioApi = makePortfolioApi();
    await boot(makeCtx(), portfolioApi, makeAttributionSvc());

    expect(portfolioApi.getLatestSnapshot.calls.count())
      .withContext('getLatestSnapshot must be called on component init when a portfolio is selected')
      .toBeGreaterThan(0);
  });

  it('when portfolio is in context, getLatestSnapshot is called with the NAME, not the UUID', async () => {
    const portfolioApi = makePortfolioApi();
    await boot(makeCtx(PORTFOLIO_NAME, PORTFOLIO_ID), portfolioApi, makeAttributionSvc());

    const firstArgs = (portfolioApi.getLatestSnapshot.calls.allArgs() as unknown[][]).map(
      (a) => a[0],
    );
    expect(firstArgs.some((arg) => arg === PORTFOLIO_NAME))
      .withContext('getLatestSnapshot must receive the resolved portfolio name')
      .toBeTrue();
    expect(firstArgs.some((arg) => arg === PORTFOLIO_ID))
      .withContext('getLatestSnapshot must never receive the raw UUID')
      .toBeFalse();
  });

  it('when portfolio is in context, brinson attribution service is called on init', async () => {
    const attrSvc = makeAttributionSvc();
    await boot(makeCtx(), makePortfolioApi(), attrSvc);

    expect(attrSvc.brinson.calls.count())
      .withContext('brinson must be called on init when portfolio and snapshot are present')
      .toBeGreaterThan(0);
  });

  it('when no portfolio is in context, getLatestSnapshot is not called', async () => {
    const portfolioApi = makePortfolioApi();
    await boot(makeCtx(null, null), portfolioApi, makeAttributionSvc());

    expect(portfolioApi.getLatestSnapshot.calls.count())
      .withContext('getLatestSnapshot must not be called when no portfolio is selected')
      .toBe(0);
  });

  it('when portfolio id is set but name has not resolved (cold boot), getLatestSnapshot is not called', async () => {
    // Simulates the window where id is restored from localStorage before the portfolio list resolves
    const portfolioApi = makePortfolioApi();
    await boot(makeCtx(null, PORTFOLIO_ID), portfolioApi, makeAttributionSvc());

    expect(portfolioApi.getLatestSnapshot.calls.count())
      .withContext(
        'getLatestSnapshot must not fire during cold-boot window when name has not yet resolved'
      )
      .toBe(0);
  });

  it('when name resolves after a cold boot, getLatestSnapshot fires with the name (not the id)', async () => {
    const portfolioApi = makePortfolioApi();
    const ctx          = makeCtx(null, PORTFOLIO_ID); // cold-boot: id known, name null
    const fixture      = await boot(ctx, portfolioApi, makeAttributionSvc());

    portfolioApi.getLatestSnapshot.calls.reset();

    ctx._setTo(PORTFOLIO_NAME, PORTFOLIO_ID);
    fixture.detectChanges();

    const firstArgs = (portfolioApi.getLatestSnapshot.calls.allArgs() as unknown[][]).map(
      (a) => a[0],
    );
    expect(firstArgs.length)
      .withContext('getLatestSnapshot must fire once name resolves')
      .toBeGreaterThan(0);
    expect(firstArgs.every((arg) => arg === PORTFOLIO_NAME))
      .withContext('All calls after name resolves must use the name, never the UUID')
      .toBeTrue();
  });
});

// ─── Suite: weights sourced from getLatestSnapshot ───────────────────────────

describe('AttributionComponent — portfolio weights sourced from getLatestSnapshot (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "Portfolio weights are sourced from PortfolioApiService.getLatestSnapshot(name),
   * not a manual portfolio-weights selection UI."
   * Criterion: "portfolioWeights is populated from getLatestSnapshot(name)."
   *
   * Assumption: the component exposes a `portfolioWeights()` signal whose value
   * is derived from the `weights` field of the snapshot response.  This name
   * is taken directly from the requirements text ("portfolioWeights").
   */

  it('when portfolio is in context, portfolioWeights() is populated from the snapshot weights', async () => {
    const fixture = await boot(makeCtx(), makePortfolioApi(), makeAttributionSvc());
    const comp    = fixture.componentInstance;

    expect(comp.portfolioWeights())
      .withContext('portfolioWeights must be populated from getLatestSnapshot(name) response')
      .toEqual(jasmine.objectContaining({ AAPL: 0.5, MSFT: 0.5 }));
  });

  it('when snapshot has empty weights, portfolioWeights() is an empty object', async () => {
    const emptySnapshotApi: PortfolioApiMock = {
      getLatestSnapshot: jasmine
        .createSpy('getLatestSnapshot')
        .and.returnValue(
          of({ id: '2', portfolioName: PORTFOLIO_NAME, date: '2025-01-01', weights: {} }),
        ),
      listPortfolios: jasmine.createSpy('listPortfolios').and.returnValue(of({ items: [] })),
    };
    const fixture = await boot(makeCtx(), emptySnapshotApi, makeAttributionSvc());
    const comp    = fixture.componentInstance;

    expect(comp.portfolioWeights())
      .withContext('portfolioWeights must be empty when the snapshot response has no weights')
      .toEqual({});
  });

  // Invariant: for all valid portfolio names, getLatestSnapshot always receives the name
  // (never the id) — tested across a range of name values.
  it('when a variety of portfolio names are set in context, getLatestSnapshot always receives the name', async () => {
    const testPairs: Array<[string, string]> = [
      ['alpha-fund', 'ffffffff-0000-0000-0000-000000000001'],
      ['My Portfolio', 'ffffffff-0000-0000-0000-000000000002'],
      ['t212', PORTFOLIO_ID],
    ];

    // Start with null so every _setTo call is a genuine signal change that fires the effect.
    const ctx = makeCtx(null, null);
    const portfolioApi = makePortfolioApi();
    const fixture = await boot(ctx, portfolioApi, makeAttributionSvc());

    for (const [name, id] of testPairs) {
      portfolioApi.getLatestSnapshot.calls.reset();
      ctx._setTo(name, id);
      fixture.detectChanges();

      const firstArgs = (portfolioApi.getLatestSnapshot.calls.allArgs() as unknown[][]).map(
        (a) => a[0],
      );
      expect(firstArgs.some((arg) => arg === name))
        .withContext(`For name "${name}", getLatestSnapshot must be called with the name`)
        .toBeTrue();
      expect(firstArgs.some((arg) => arg === id))
        .withContext(`For name "${name}" / id "${id}", getLatestSnapshot must never receive the UUID`)
        .toBeFalse();
    }
  });
});

// ─── Suite: (b) handles API error ────────────────────────────────────────────

describe('AttributionComponent — (b) handles API error (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "Every page/panel shows a visible, non-blank error state (message or
   * alert component) when its primary API call returns an error."
   * Criterion: "Every wired page has specs covering: (b) handles API error."
   */

  it('when getLatestSnapshot throws, a role="alert" element is present in the DOM', async () => {
    const fixture = await boot(
      makeCtx(),
      makePortfolioApi({ snapshotError: true }),
      makeAttributionSvc(),
    );
    fixture.detectChanges();

    const el      = fixture.nativeElement as HTMLElement;
    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl)
      .withContext('A visible error element must appear when getLatestSnapshot fails')
      .not.toBeNull();
  });

  it('when getLatestSnapshot throws, the error element contains non-blank text', async () => {
    const fixture = await boot(
      makeCtx(),
      makePortfolioApi({ snapshotError: true }),
      makeAttributionSvc(),
    );
    fixture.detectChanges();

    const el      = fixture.nativeElement as HTMLElement;
    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    const text = alertEl?.textContent?.trim() ?? '';
    expect(text.length)
      .withContext('Error element must contain non-blank text')
      .toBeGreaterThan(0);
  });

  it('when brinson API throws, a role="alert" element is present in the DOM', async () => {
    const fixture = await boot(
      makeCtx(),
      makePortfolioApi(),
      makeAttributionSvc({ brinsonError: true }),
    );
    fixture.detectChanges();

    const el      = fixture.nativeElement as HTMLElement;
    const alertEl =
      el.querySelector('[role="alert"]') ??
      el.querySelector('app-page-error-banner') ??
      el.querySelector('app-alert-banner');

    expect(alertEl)
      .withContext('A visible error element must appear when the brinson attribution call fails')
      .not.toBeNull();
  });
});

// ─── Suite: no unhandled console.error on API failure ────────────────────────

describe('AttributionComponent — no unhandled observable errors reach console.error (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "No unhandled observable errors reach console.error."
   */

  it('when snapshot and all attribution APIs throw, console.error is not called', async () => {
    const consoleErrorSpy = spyOn(console, 'error');

    const fixture = await boot(
      makeCtx(),
      makePortfolioApi({ snapshotError: true }),
      makeAttributionSvc({ brinsonError: true, factorError: true }),
    );
    fixture.detectChanges();

    expect(consoleErrorSpy)
      .withContext(
        'catchError must prevent unhandled observable errors from reaching console.error'
      )
      .not.toHaveBeenCalled();
  });

  it('when only brinson throws, console.error is not called', async () => {
    const consoleErrorSpy = spyOn(console, 'error');

    const fixture = await boot(
      makeCtx(),
      makePortfolioApi(),
      makeAttributionSvc({ brinsonError: true }),
    );
    fixture.detectChanges();

    expect(consoleErrorSpy)
      .withContext('brinson error must be caught — must not reach console.error')
      .not.toHaveBeenCalled();
  });
});

// ─── Suite: (c) reacts to portfolio context change ────────────────────────────

describe('AttributionComponent — (c) reacts to portfolio context change (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "Every wired page has specs covering: (c) reacts to portfolio context change."
   * Criterion: "switching the portfolio refetches the snapshot with the new name."
   */

  let ctx: CtxMock;
  let portfolioApi: PortfolioApiMock;
  let attrSvc: AttributionSvcMock;
  let fixture: ComponentFixture<AttributionComponent>;

  beforeEach(async () => {
    ctx          = makeCtx();
    portfolioApi = makePortfolioApi();
    attrSvc      = makeAttributionSvc();
    fixture      = await boot(ctx, portfolioApi, attrSvc);
    // Isolate: reset counts so only post-switch calls are measured
    portfolioApi.getLatestSnapshot.calls.reset();
    attrSvc.brinson.calls.reset();
    attrSvc.factor.calls.reset();
  });

  it('when the portfolio name changes, getLatestSnapshot is called again', () => {
    ctx._setTo(SECOND_NAME, SECOND_ID);
    fixture.detectChanges();

    expect(portfolioApi.getLatestSnapshot.calls.count())
      .withContext('getLatestSnapshot must be re-called after portfolio switch')
      .toBeGreaterThan(0);
  });

  it('when the portfolio name changes, getLatestSnapshot is called with the NEW name', () => {
    ctx._setTo(SECOND_NAME, SECOND_ID);
    fixture.detectChanges();

    const firstArgs = (portfolioApi.getLatestSnapshot.calls.allArgs() as unknown[][]).map(
      (a) => a[0],
    );
    expect(firstArgs.some((arg) => arg === SECOND_NAME))
      .withContext(`getLatestSnapshot must use '${SECOND_NAME}' after portfolio switch`)
      .toBeTrue();
  });

  it('when the portfolio name changes, getLatestSnapshot is NOT called with the old name', () => {
    ctx._setTo(SECOND_NAME, SECOND_ID);
    fixture.detectChanges();

    const firstArgs = (portfolioApi.getLatestSnapshot.calls.allArgs() as unknown[][]).map(
      (a) => a[0],
    );
    expect(firstArgs.some((arg) => arg === PORTFOLIO_NAME))
      .withContext('getLatestSnapshot must not re-use the old portfolio name after switch')
      .toBeFalse();
  });

  it('when portfolio is cleared to null, no new API calls are made', () => {
    ctx._setTo(null, null);
    fixture.detectChanges();

    const totalCalls =
      portfolioApi.getLatestSnapshot.calls.count() +
      attrSvc.brinson.calls.count() +
      attrSvc.factor.calls.count();
    expect(totalCalls)
      .withContext('No API calls should fire when portfolio is cleared to null')
      .toBe(0);
  });
});

// ─── Suite: context service is the sole source of portfolio identity ──────────

describe('AttributionComponent — context service is the sole source of portfolio identity (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "attribution reads portfolioContextService.selectedPortfolio() /
   * currentPortfolioName()."
   *
   * Behavioural proxy: changing only context signals triggers a refetch.  If the
   * component maintained a divergent local writable signal, the signal-only update
   * would not be enough to trigger the refetch, and this test would fail.
   */

  it('when only context signals change, the component refetches snapshot without any local-state mutation', async () => {
    const ctx          = makeCtx();
    const portfolioApi = makePortfolioApi();
    const fixture      = await boot(ctx, portfolioApi, makeAttributionSvc());

    portfolioApi.getLatestSnapshot.calls.reset();

    // Change ONLY via the context signals — no component method called directly
    ctx._setTo(SECOND_NAME, SECOND_ID);
    fixture.detectChanges();

    const firstArgs = (portfolioApi.getLatestSnapshot.calls.allArgs() as unknown[][]).map(
      (a) => a[0],
    );
    expect(firstArgs.some((arg) => arg === SECOND_NAME))
      .withContext('Component must react to context signal change — no local-state shim required')
      .toBeTrue();
  });

  it('when portfolio context resolves to null, the component makes no API calls regardless of prior state', async () => {
    const ctx          = makeCtx();
    const portfolioApi = makePortfolioApi();
    const attrSvc      = makeAttributionSvc();
    const fixture      = await boot(ctx, portfolioApi, attrSvc);

    portfolioApi.getLatestSnapshot.calls.reset();
    attrSvc.brinson.calls.reset();
    attrSvc.factor.calls.reset();

    ctx._setTo(null, null);
    fixture.detectChanges();

    const totalCalls =
      portfolioApi.getLatestSnapshot.calls.count() +
      attrSvc.brinson.calls.count() +
      attrSvc.factor.calls.count();
    expect(totalCalls)
      .withContext('Clearing context must not trigger any attribution API call')
      .toBe(0);
  });
});

// ─── Suite: no API endpoint names in user-facing copy ────────────────────────

describe('AttributionComponent — no API endpoint or parameter names in user-facing copy (issue #1017 [UNIT])', () => {
  /**
   * Criterion: "No API endpoint or parameter names appear in any user-facing UI copy."
   *
   * The attribution page previously called GET /attribution/brinson/{name} and
   * GET /attribution/factor/{name} (wrong endpoints now fixed).  This test guards
   * against endpoint paths or HTTP method names leaking into the rendered text.
   */

  let fixture: ComponentFixture<AttributionComponent>;

  beforeEach(async () => {
    fixture = await boot(makeCtx(), makePortfolioApi(), makeAttributionSvc());
    fixture.detectChanges();
  });

  it('when attribution renders, the text does not contain the brinson endpoint path', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text)
      .withContext('/attribution/brinson is an internal API path and must not appear in UI copy')
      .not.toContain('/attribution/brinson');
  });

  it('when attribution renders, the text does not contain the factor endpoint path', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text)
      .withContext('/attribution/factor is an internal API path and must not appear in UI copy')
      .not.toContain('/attribution/factor');
  });

  it('when attribution renders, the text does not contain a raw HTTP method name like "POST /"', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text)
      .withContext('"POST /" is an implementation detail and must not appear in UI copy')
      .not.toContain('POST /');
  });
});
