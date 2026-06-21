/**
 * Source-blind spec — issue #1030, scope criterion 12
 *
 * Criterion: Total Return, Sharpe, Max Drawdown, and Information Ratio render
 * "—" (em-dash) before any run is executed, and a non-empty formatted result
 * after a run completes (locked by spec).
 *
 * Written against the acceptance criteria only; no implementation source was
 * read. Tests are in the Red phase and will fail until the implementation is
 * complete.
 *
 * Implementation contract implied by these tests:
 *   • BacktestingComponent renders four KPI value elements addressable by
 *     data-testid="kpi-total-return|kpi-sharpe-ratio|kpi-max-drawdown|kpi-information-ratio"
 *   • Each element's text content is exactly "—" when no run result has been loaded
 *   • BacktestService exposes submit(params) → Observable<{jobId}> and
 *     getResult(jobId) → Observable<{status, result?}> observable streams
 *   • A run button with data-testid="run-backtest-btn" triggers the submit→poll flow
 *
 * Note: import path uses ./backtesting (no .component suffix) to match the
 * project's barrel export convention observed in sibling spec files.
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection, signal, WritableSignal } from '@angular/core';
import { Subject } from 'rxjs';

import { BacktestingComponent } from './backtesting';
import { BacktestService } from './backtest.service';
import { ICON_PROVIDER } from '../icons';
import {
  PortfolioContextService,
  type DateRange,
} from '../core/services/portfolio-context.service';

const EM_DASH = '—';

interface MockPortfolioContext {
  currentPortfolioId: WritableSignal<string | null>;
  currentPortfolioName: WritableSignal<string | null>;
  selectedPortfolio: WritableSignal<{ id: string; name: string } | null>;
  dateRange: WritableSignal<DateRange>;
}

function makeCtxStub(portfolioId: string | null = null): MockPortfolioContext {
  return {
    currentPortfolioId: signal<string | null>(portfolioId),
    currentPortfolioName: signal<string | null>(portfolioId ? 't212' : null),
    selectedPortfolio: signal<{ id: string; name: string } | null>(
      portfolioId ? { id: portfolioId, name: 't212' } : null,
    ),
    dateRange: signal<DateRange>({
      preset: '1Y',
      start: new Date('2024-01-01'),
      end: new Date('2025-01-01'),
    }),
  };
}

const MOCK_COMPLETED_RESULT = {
  status: 'completed' as const,
  result: {
    totalReturn: 0.1532,
    sharpeRatio: 1.42,
    maxDrawdown: -0.0812,
    informationRatio: 0.87,
    equityCurve: {
      dates: ['2023-01-01', '2023-12-31'],
      values: [100, 115.32],
    },
  },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function kpi(fixture: ComponentFixture<BacktestingComponent>, testId: string): HTMLElement | null {
  return fixture.nativeElement.querySelector(`[data-testid="${testId}"]`);
}

// ─── Pre-run: all four KPIs must show "—" ────────────────────────────────────

describe('BacktestingComponent — KPI em-dash before any run (issue #1030, criterion 12)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: PortfolioContextService, useValue: makeCtxStub() },
        ICON_PROVIDER,
      ],
    });
    fixture = TestBed.createComponent(BacktestingComponent);
    http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    http.match(() => true);
  });

  afterEach(() => http.verify());

  it('when no run has been executed, Total Return KPI shows em-dash', () => {
    const el = kpi(fixture, 'kpi-total-return');
    expect(el)
      .withContext('[data-testid="kpi-total-return"] must exist in the template')
      .not.toBeNull();
    expect(el!.textContent!.trim()).toBe(EM_DASH);
  });

  it('when no run has been executed, Sharpe Ratio KPI shows em-dash', () => {
    const el = kpi(fixture, 'kpi-sharpe-ratio');
    expect(el)
      .withContext('[data-testid="kpi-sharpe-ratio"] must exist in the template')
      .not.toBeNull();
    expect(el!.textContent!.trim()).toBe(EM_DASH);
  });

  it('when no run has been executed, Max Drawdown KPI shows em-dash', () => {
    const el = kpi(fixture, 'kpi-max-drawdown');
    expect(el)
      .withContext('[data-testid="kpi-max-drawdown"] must exist in the template')
      .not.toBeNull();
    expect(el!.textContent!.trim()).toBe(EM_DASH);
  });

  it('when no run has been executed, Information Ratio KPI shows em-dash', () => {
    const el = kpi(fixture, 'kpi-information-ratio');
    expect(el)
      .withContext('[data-testid="kpi-information-ratio"] must exist in the template')
      .not.toBeNull();
    expect(el!.textContent!.trim()).toBe(EM_DASH);
  });
});

// ─── Post-run: KPIs update from "—" to formatted values ──────────────────────

describe('BacktestingComponent — KPI formatted values after run completes (issue #1030, criterion 12)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let backtestSpy: any;
  let http: HttpTestingController;
  let pollResult$: Subject<typeof MOCK_COMPLETED_RESULT>;

  beforeEach(() => {
    pollResult$ = new Subject();
    backtestSpy = jasmine.createSpyObj('BacktestService', [
      'submit',
      'getResult',
    ]);
    backtestSpy.getResult.and.returnValue(pollResult$.asObservable());

    TestBed.configureTestingModule({
      imports: [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: PortfolioContextService, useValue: makeCtxStub('port-1') },
        { provide: BacktestService, useValue: backtestSpy },
        ICON_PROVIDER,
      ],
    });
    fixture = TestBed.createComponent(BacktestingComponent);
    http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    http.match(() => true);

    // Deliver a completed result via the poll observable
    pollResult$.next(MOCK_COMPLETED_RESULT);
    fixture.detectChanges();
  });

  afterEach(() => {
    pollResult$.complete();
    http.verify();
  });

  it('when a run result arrives, Total Return KPI shows a non-empty value other than em-dash', () => {
    const el = kpi(fixture, 'kpi-total-return');
    expect(el).not.toBeNull();
    const text = el!.textContent!.trim();
    expect(text).not.toBe(EM_DASH);
    expect(text.length).toBeGreaterThan(0);
  });

  it('when a run result arrives, Sharpe Ratio KPI shows a non-empty value other than em-dash', () => {
    const el = kpi(fixture, 'kpi-sharpe-ratio');
    expect(el).not.toBeNull();
    const text = el!.textContent!.trim();
    expect(text).not.toBe(EM_DASH);
    expect(text.length).toBeGreaterThan(0);
  });

  it('when a run result arrives, Max Drawdown KPI shows a non-empty value other than em-dash', () => {
    const el = kpi(fixture, 'kpi-max-drawdown');
    expect(el).not.toBeNull();
    const text = el!.textContent!.trim();
    expect(text).not.toBe(EM_DASH);
    expect(text.length).toBeGreaterThan(0);
  });

  it('when a run result arrives, Information Ratio KPI shows a non-empty value other than em-dash', () => {
    const el = kpi(fixture, 'kpi-information-ratio');
    expect(el).not.toBeNull();
    const text = el!.textContent!.trim();
    expect(text).not.toBe(EM_DASH);
    expect(text.length).toBeGreaterThan(0);
  });
});
