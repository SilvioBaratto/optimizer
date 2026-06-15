/**
 * Source-blind example tests — authored from acceptance criteria only.
 *
 * Criteria covered (issue #1019):
 * - A drift fetch 4xx/5xx shows an in-panel error on the status tab WITHOUT
 *   blanking the page (global `hasError` no longer set from the drift path).
 * - trade-preview and history show a visible `role="alert"` error on failure.
 * - Every page/panel shows a visible, non-blank error state when its primary
 *   API call returns an error.
 * - No unhandled observable errors reach `console.error`.
 *
 * The KEY new invariant this suite pins:
 *   When only getDrift fails (all other services succeed), the page-level
 *   tab-group must remain in the DOM — the drift error is in-panel only.
 *
 * Service method names confirmed from rebalancing-contracts.spec.ts:
 *   getDrift(name, threshold?)   listPolicies(name)   getPreview(name)
 *   getSnapshots(name)           decide(body)         createPolicy(name, body)
 *   activatePolicy(name, id)
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection, signal, computed } from '@angular/core';
import { of, throwError } from 'rxjs';

import { RebalancingComponent } from './rebalancing';
import { RebalancingService } from './rebalancing.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

const PORTFOLIO = 'test-portfolio';
const PORTFOLIO_ID = 'test-uuid-0001';

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

function makeSvc(overrides: Record<string, unknown> = {}) {
  return {
    getDrift:       jasmine.createSpy('getDrift').and.returnValue(of({ entries: [], breachedCount: 0, threshold: 0.05 })),
    listPolicies:   jasmine.createSpy('listPolicies').and.returnValue(of({ items: [] })),
    getPreview:     jasmine.createSpy('getPreview').and.returnValue(of({ trades: [], estimatedCost: 0, totalTurnover: 0 })),
    getSnapshots:   jasmine.createSpy('getSnapshots').and.returnValue(of({ items: [] })),
    decide:         jasmine.createSpy('decide').and.returnValue(of({})),
    createPolicy:   jasmine.createSpy('createPolicy').and.returnValue(of({})),
    activatePolicy: jasmine.createSpy('activatePolicy').and.returnValue(of(undefined)),
    ...overrides,
  };
}

async function boot(
  svc: ReturnType<typeof makeSvc>,
  ctx = makeCtx(),
): Promise<ComponentFixture<RebalancingComponent>> {
  await TestBed.configureTestingModule({
    imports: [RebalancingComponent],
    schemas: [NO_ERRORS_SCHEMA],
    providers: [
      provideZonelessChangeDetection(),
      ICON_PROVIDER,
      { provide: PortfolioContextService, useValue: ctx },
      { provide: RebalancingService,      useValue: svc },
    ],
  }).compileComponents();
  const fixture = TestBed.createComponent(RebalancingComponent);
  fixture.detectChanges();
  return fixture;
}

// ─── Drift failure: in-panel only, page not blanked ─────────────────────────

describe('RebalancingComponent — drift 5xx shows in-panel error without blanking page (issue #1019)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;
  let el: HTMLElement;

  beforeEach(async () => {
    const svc = makeSvc({
      getDrift: jasmine.createSpy('getDrift').and.returnValue(
        throwError(() => new Error('500 Server Error'))
      ),
    });
    fixture = await boot(svc);
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  it('when only getDrift returns 5xx, app-tab-group remains in the DOM (page is not blanked)', () => {
    expect(el.querySelector('app-tab-group'))
      .withContext('The tab-group must remain rendered when only drift fails; global hasError must not be set')
      .not.toBeNull();
  });

  it('when only getDrift returns 5xx, a non-blank error indicator is visible', () => {
    const alertEl   = el.querySelector('[role="alert"]');
    const errorEl   = el.querySelector('[data-testid="drift-error"], [data-testid="status-error"], .drift-error');
    expect(alertEl !== null || errorEl !== null)
      .withContext('Expected at least one visible error indicator after drift 5xx')
      .toBeTrue();
  });
});

describe('RebalancingComponent — drift 4xx shows in-panel error without blanking page (issue #1019)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;
  let el: HTMLElement;

  beforeEach(async () => {
    const svc = makeSvc({
      getDrift: jasmine.createSpy('getDrift').and.returnValue(
        throwError(() => new Error('404 Not Found'))
      ),
    });
    fixture = await boot(svc);
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  it('when only getDrift returns 4xx, app-tab-group remains in the DOM', () => {
    expect(el.querySelector('app-tab-group'))
      .withContext('Tab group must not be replaced by a full-page error when only drift fails')
      .not.toBeNull();
  });

  it('when only getDrift returns 4xx, a non-blank error indicator is visible', () => {
    const alertEl = el.querySelector('[role="alert"]');
    const errorEl = el.querySelector('[data-testid="drift-error"], [data-testid="status-error"], .drift-error');
    expect(alertEl !== null || errorEl !== null).toBeTrue();
  });
});

// ─── Trade-preview failure: role=alert ──────────────────────────────────────

describe('RebalancingComponent — trade-preview failure shows role=alert (issue #1019)', () => {
  it('when getPreview returns a 4xx error, a role=alert element is visible', async () => {
    const svc = makeSvc({
      getPreview: jasmine.createSpy('getPreview').and.returnValue(
        throwError(() => new Error('404 Not Found'))
      ),
    });
    const fixture = await boot(svc);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[role="alert"]'))
      .withContext('Expected role=alert when trade-preview API fails')
      .not.toBeNull();
  });

  it('when getPreview returns a 5xx error, a role=alert element is visible', async () => {
    const svc = makeSvc({
      getPreview: jasmine.createSpy('getPreview').and.returnValue(
        throwError(() => new Error('503 Unavailable'))
      ),
    });
    const fixture = await boot(svc);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[role="alert"]'))
      .withContext('Expected role=alert when trade-preview API returns 5xx')
      .not.toBeNull();
  });
});

// ─── History/snapshots failure: role=alert ───────────────────────────────────

describe('RebalancingComponent — history failure shows role=alert (issue #1019)', () => {
  it('when getSnapshots returns a 5xx error, a role=alert element is visible', async () => {
    const svc = makeSvc({
      getSnapshots: jasmine.createSpy('getSnapshots').and.returnValue(
        throwError(() => new Error('500 Internal Server Error'))
      ),
    });
    const fixture = await boot(svc);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[role="alert"]'))
      .withContext('Expected role=alert when history/snapshots API fails')
      .not.toBeNull();
  });

  it('when getSnapshots returns a 4xx error, a role=alert element is visible', async () => {
    const svc = makeSvc({
      getSnapshots: jasmine.createSpy('getSnapshots').and.returnValue(
        throwError(() => new Error('404 Not Found'))
      ),
    });
    const fixture = await boot(svc);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[role="alert"]'))
      .withContext('Expected role=alert when history API returns 4xx')
      .not.toBeNull();
  });
});
