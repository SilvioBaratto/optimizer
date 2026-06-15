/**
 * Source-blind example tests — authored from acceptance criteria only.
 *
 * Criterion covered (global, issue #1019):
 * - No API endpoint or parameter names appear in any user-facing UI copy.
 *
 * Requirements.md scope 13e calls out a specific instance:
 *   The walk-forward section description must not contain the text
 *   "POST /validate/walk-forward" or "cv_type=walk_forward".
 *
 * For the rebalancing page, the rendered text must not expose raw endpoint
 * strings such as /rebalance/decide, /rebalance-policy, /rebalance/preview,
 * /portfolio-analytics, etc.
 *
 * This suite is intentionally independent of the other rebalancing specs:
 * it mounts the component with happy-path service stubs and asserts that
 * none of the forbidden patterns appear in the textContent of the rendered DOM.
 */

import { TestBed, ComponentFixture } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection, signal, computed } from '@angular/core';
import { of } from 'rxjs';

import { RebalancingComponent } from './rebalancing';
import { RebalancingService } from './rebalancing.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

// ─── Forbidden patterns: raw API endpoint or parameter strings ───────────────
// Derived directly from the acceptance criterion and requirements.md scope 13e.

const FORBIDDEN_API_PATTERNS: ReadonlyArray<{ label: string; pattern: string }> = [
  { label: 'rebalance decide endpoint',        pattern: '/rebalance/decide'       },
  { label: 'rebalance-policy path segment',    pattern: 'rebalance-policy'        },
  { label: 'rebalance preview endpoint',       pattern: '/rebalance/preview'      },
  { label: 'portfolio-analytics path segment', pattern: '/portfolio-analytics'    },
  { label: 'walk-forward POST description',    pattern: 'POST /validate'          },
  { label: 'cv_type parameter name',           pattern: 'cv_type='               },
  { label: 'raw {name} placeholder',           pattern: '{name}'                  },
  { label: 'raw {portfolio_name} placeholder', pattern: '{portfolio_name}'        },
];

function makeCtx(name = 'test-portfolio', id = 'test-uuid') {
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

function makeHappyPathSvc() {
  return {
    getDrift:       jasmine.createSpy('getDrift').and.returnValue(of({ entries: [], breachedCount: 0, threshold: 0.05 })),
    listPolicies:   jasmine.createSpy('listPolicies').and.returnValue(of({ items: [] })),
    getPreview:     jasmine.createSpy('getPreview').and.returnValue(of({ trades: [], estimatedCost: 0, totalTurnover: 0 })),
    getSnapshots:   jasmine.createSpy('getSnapshots').and.returnValue(of({ items: [] })),
    decide:         jasmine.createSpy('decide').and.returnValue(of({})),
    createPolicy:   jasmine.createSpy('createPolicy').and.returnValue(of({})),
    activatePolicy: jasmine.createSpy('activatePolicy').and.returnValue(of(undefined)),
  };
}

describe('RebalancingComponent — no API endpoint names in user-facing copy (issue #1019)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;
  let pageText: string;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RebalancingComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtx() },
        { provide: RebalancingService,      useValue: makeHappyPathSvc() },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(RebalancingComponent);
    fixture.detectChanges();
    pageText = (fixture.nativeElement as HTMLElement).textContent ?? '';
  });

  FORBIDDEN_API_PATTERNS.forEach(({ label, pattern }) => {
    it(`when the rebalancing page renders, the ${label} ("${pattern}") does not appear in user-facing text`, () => {
      expect(pageText).not.toContain(
        pattern,
        `User-facing copy must not expose the internal API detail: "${pattern}"`
      );
    });
  });

  it('when the rebalancing page renders, no raw HTTP method + path combos appear in visible text', () => {
    // Covers any "POST /..." or "GET /..." pattern that would expose internal wiring
    expect(pageText).not.toMatch(
      /\b(GET|POST|PUT|DELETE|PATCH)\s+\/[a-z]/,
      'User-facing copy must not contain raw HTTP method + path strings'
    );
  });
});
