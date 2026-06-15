/**
 * Source-blind parity lock for issue #1018.
 *
 * Criterion: [UNIT] `PerformanceMetrics` field-by-field parity vs the
 * GET /portfolio-analytics/{name}/performance-metrics endpoint is asserted
 * (lock, not the Cycle-5 cross-page contract suite).
 *
 * Fields under lock (from issue #1018 acceptance criteria):
 *   Response top-level:  kpis[], nav, navChangePct, currency
 *   Each kpi item:       label, value, format, change, changeLabel, sparkline
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

import { DashboardService } from './dashboard.service';
import type { ApiPerformanceMetricsResponse } from '../core/models/dashboard-api.model';

// ── Canonical response fixture ────────────────────────────────────────────────

const CANONICAL_PERF_METRICS_RESPONSE: ApiPerformanceMetricsResponse = {
  kpis: [
    {
      label: 'Sharpe Ratio',
      value: 1.42,
      format: 'ratio',
      change: 0.05,
      changeLabel: '+5 bps',
      sparkline: [1.2, 1.3, 1.35, 1.4, 1.42],
    },
    {
      label: 'Total Return',
      value: 0.123,
      format: 'percent',
      change: 0.01,
      changeLabel: '+1%',
      sparkline: [0.08, 0.09, 0.1, 0.11, 0.123],
    },
  ],
  nav: 250_000,
  navChangePct: 0.0123,
  currency: 'EUR',
};

// ── Helper ────────────────────────────────────────────────────────────────────

function callAndFlush(
  svc: DashboardService,
  http: HttpTestingController,
  cb: (v: ApiPerformanceMetricsResponse) => void,
): void {
  svc.getPerformanceMetrics('fund', '1Y').subscribe(cb);
  http.expectOne((r) => r.url.includes('performance-metrics'))
    .flush(CANONICAL_PERF_METRICS_RESPONSE);
}

// ── Suite ─────────────────────────────────────────────────────────────────────

describe('DashboardService — PerformanceMetrics field parity (issue #1018)', () => {
  let svc: DashboardService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        DashboardService,
      ],
    });
    svc = TestBed.inject(DashboardService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  // ── Top-level shape ───────────────────────────────────────────────────────

  it('when getPerformanceMetrics returns a response, the emitted value has a kpis array', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(Array.isArray(result?.kpis)).toBeTrue();
  });

  it('when getPerformanceMetrics returns a response, the emitted value has nav as a number', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.nav).toBe('number');
  });

  it('when getPerformanceMetrics returns a response, the emitted value has navChangePct as a number', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.navChangePct).toBe('number');
  });

  it('when getPerformanceMetrics returns a response, the emitted value has currency as a string', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.currency).toBe('string');
  });

  // ── KPI item shape ────────────────────────────────────────────────────────

  it('when kpis array is non-empty, each item has a label string', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.kpis[0].label).toBe('string');
  });

  it('when kpis array is non-empty, each item has a value number', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.kpis[0].value).toBe('number');
  });

  it('when kpis array is non-empty, each item has a format string', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.kpis[0].format).toBe('string');
  });

  it('when kpis array is non-empty, each item has a change number', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.kpis[0].change).toBe('number');
  });

  it('when kpis array is non-empty, each item has a changeLabel string', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(typeof result?.kpis[0].changeLabel).toBe('string');
  });

  it('when kpis array is non-empty, each item has a sparkline array', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect(Array.isArray(result?.kpis[0].sparkline)).toBeTrue();
  });

  // ── Negative: no stale field names from the old contract-parity spec ──────

  it('when getPerformanceMetrics returns a response, the response has no totalReturn field', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect('totalReturn' in (result as object)).toBeFalse();
  });

  it('when getPerformanceMetrics returns a response, the response has no sharpeRatio field', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });
    expect('sharpeRatio' in (result as object)).toBeFalse();
  });

  // ── Full shape snapshot ───────────────────────────────────────────────────

  it('when getPerformanceMetrics returns a two-kpi response, all required fields are present', () => {
    let result: ApiPerformanceMetricsResponse | undefined;
    callAndFlush(svc, http, (v) => { result = v; });

    expect(result).toEqual(
      jasmine.objectContaining({
        kpis: jasmine.arrayContaining([
          jasmine.objectContaining({
            label: jasmine.any(String),
            value: jasmine.any(Number),
            format: jasmine.any(String),
            change: jasmine.any(Number),
            changeLabel: jasmine.any(String),
            sparkline: jasmine.any(Array),
          }),
        ]),
        nav: jasmine.any(Number),
        navChangePct: jasmine.any(Number),
        currency: jasmine.any(String),
      }),
    );
  });
});
