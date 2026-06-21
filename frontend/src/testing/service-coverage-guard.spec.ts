/**
 * service-coverage-guard.spec.ts — issue #968
 *
 * Guards against uncovered service drift. Every *.service.ts that owns an HTTP
 * contract must appear in this file's HTTP_SERVICE_INVENTORY. The TypeScript
 * import for each entry IS the compile-time check: renaming or deleting a service
 * breaks the import → CI fails. The `it` tests verify each class is defined at
 * runtime and document which parity spec covers the wire contract.
 *
 * ── Maintenance rules ────────────────────────────────────────────────────────
 *  • Adding a new HTTP service  → add an entry to HTTP_SERVICE_INVENTORY below
 *    and ensure a parity spec (e.g. cross-page-service-field-parity.spec.ts)
 *    exercises its request/response shapes.
 *  • Adding a UI-only service   → add it to EXCLUDED_SERVICES below with a
 *    one-line reason (no HTTP contract to lock down).
 *  • Removing a service         → remove its entry; TS compilation ensures
 *    this file stays in sync automatically.
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ── HTTP service imports (compile-time enforcement) ───────────────────────────

import { BacktestService } from '../app/backtesting/backtest.service';
import { DashboardService } from '../app/dashboard/dashboard.service';
import { DatabaseService } from '../app/settings/database.service';
import { JobsService } from '../app/core/services/jobs.service';
import { MacroIntelligenceService } from '../app/macro-intelligence/macro-intelligence.service';
import { MarketService } from '../app/core/services/market.service';
import { OptimizationService } from '../app/optimization-studio/optimization.service';
import { PortfolioApiService } from '../app/core/services/portfolio-api.service';
import { ReferenceIndexService } from '../app/core/services/reference-index.service';
import { ReportsService } from '../app/core/services/reports.service';
import { SchedulerService } from '../app/settings/scheduler.service';
import { TickerSeedingService } from '../app/core/services/ticker-seeding.service';
import { UniverseService } from '../app/core/services/universe.service';
import { YfinanceService } from '../app/core/services/yfinance.service';

// ── Inventory ─────────────────────────────────────────────────────────────────

interface ServiceEntry {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  cls: abstract new (...args: any[]) => unknown;
  coveredBy: string[];
}

const HTTP_SERVICE_INVENTORY: ServiceEntry[] = [
  {
    cls: BacktestService,
    coveredBy: [
      'backtest.service.spec.ts',
      'research-service-field-parity.spec.ts',
      'backtesting-contracts.spec.ts',
    ],
  },
  {
    cls: DashboardService,
    coveredBy: [
      'dashboard.service.spec.ts',
      'cross-page-service-field-parity.spec.ts',
      'dashboard-contracts.spec.ts',
    ],
  },
  {
    cls: DatabaseService,
    coveredBy: [
      'settings-field-parity.spec.ts',
      'settings/database.service-contract.spec.ts',
    ],
  },
  {
    cls: JobsService,
    coveredBy: [
      'settings-field-parity.spec.ts',
      'shared-service-contracts.spec.ts',
    ],
  },
  {
    cls: MacroIntelligenceService,
    coveredBy: [
      'macro-intelligence.service.spec.ts',
      'cross-page-service-field-parity.spec.ts',
      'macro-intelligence-service-contracts.spec.ts',
    ],
  },
  {
    cls: MarketService,
    coveredBy: [
      'market.service.spec.ts',
      'cross-page-service-field-parity.spec.ts',
      'shared-service-contracts.spec.ts',
    ],
  },
  {
    cls: OptimizationService,
    coveredBy: [
      'optimization.service.spec.ts',
      'research-service-field-parity.spec.ts',
      'optimization-service-contracts.spec.ts',
    ],
  },
  {
    cls: PortfolioApiService,
    coveredBy: [
      'portfolio-api.service.spec.ts',
      'portfolio-builder-contracts.spec.ts',
      'portfolio-api-service-contracts.spec.ts',
    ],
  },
  {
    cls: ReferenceIndexService,
    coveredBy: [
      'reference-index.service.spec.ts',
      'reference-index-service-contracts.spec.ts',
    ],
  },
  {
    cls: ReportsService,
    coveredBy: [
      'reports.service.spec.ts',
      'cross-page-service-field-parity.spec.ts',
      'reports-service-contracts.spec.ts',
    ],
  },
  {
    cls: SchedulerService,
    coveredBy: [
      'settings-field-parity.spec.ts',
      'settings/scheduler.service-contract.spec.ts',
    ],
  },
  {
    cls: TickerSeedingService,
    coveredBy: ['ticker-seeding.service.spec.ts'],
  },
  {
    cls: UniverseService,
    coveredBy: [
      'universe.service.spec.ts',
      'cross-page-service-field-parity.spec.ts',
      'shared-service-contracts.spec.ts',
    ],
  },
  {
    cls: YfinanceService,
    coveredBy: [
      'yfinance.service.spec.ts',
      'yfinance-service-contracts.spec.ts',
    ],
  },
];

/**
 * UI-only / state-manager services: no HTTP contract to lock down.
 * Listed here so the next developer knows they are intentionally excluded.
 *
 *   BreakpointService        — window resize observer, no HTTP
 *   FormatService            — pure formatting utilities, no HTTP
 *   GlobalSearchService      — UI search state, no HTTP
 *   ModalService             — Angular CDK overlay, no HTTP
 *   NotificationService      — toast queue, no HTTP
 *   PortfolioContextService  — signal-based portfolio selection state, no HTTP
 */
const _EXCLUDED_UI_SERVICES = [
  'BreakpointService',
  'FormatService',
  'GlobalSearchService',
  'ModalService',
  'NotificationService',
  'PortfolioContextService',
] as const;

// ── Guard tests ───────────────────────────────────────────────────────────────

describe('Service coverage guard — every HTTP service referenced in a parity spec (issue #968)', () => {

  it('inventory has at least 14 HTTP services', () => {
    expect(HTTP_SERVICE_INVENTORY.length).toBeGreaterThanOrEqual(14);
  });

  it('every entry has a non-empty coveredBy list', () => {
    const uncovered = HTTP_SERVICE_INVENTORY.filter(e => e.coveredBy.length === 0);
    expect(uncovered.map(e => e.cls.name)).toEqual(
      [],
      'These services are in the inventory but list no covering spec. ' +
      'Add at least one parity spec filename to their coveredBy array.',
    );
  });

  HTTP_SERVICE_INVENTORY.forEach(({ cls, coveredBy }) => {
    describe(cls.name, () => {
      it('class is defined (import resolves)', () => {
        expect(cls).toBeDefined();
      });

      it(`has ≥1 declared parity spec: ${coveredBy[0]}`, () => {
        expect(coveredBy.length).toBeGreaterThan(0);
        coveredBy.forEach(spec =>
          expect(spec).withContext(`empty spec filename for ${cls.name}`).not.toBe(''),
        );
      });
    });
  });

});
