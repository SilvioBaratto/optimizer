/**
 * Criterion [UNIT]: `BuilderStore` propagates the global portfolio name into
 * name-dependent calls.
 *
 * Chain under test:
 *   store.setPortfolioName(name) →  store.portfolioName() signal
 *     → BuilderDriftService._guardsPass() checks the signal
 *     → BuilderDriftService._fetch() embeds the name in the drift URL
 *
 * These tests hold BuilderStore + BuilderDriftService together and verify
 * that the URL produced for the drift request uses whatever value was stored
 * by setPortfolioName().  The complementary URL-encoding cases live in
 * builder-drift-service-name-url.spec.ts; this file focuses on the
 * store → service coupling: name propagation, name change, and null guard.
 */

import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { EMPTY } from 'rxjs';

import { BuilderStore } from './state/builder.store';
import { BuilderDriftService } from './builder-drift.service';
import { PipelineBuilderApiService } from '../core/services/pipeline-builder-api.service';

const UUID_RE =
  /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;

function minimalDriftBody() {
  return {
    drift: [],
    trades: [],
    holdings: [],
    target: [],
    totals: {
      deployable_eur: 0,
      total_holdings_eur: 0,
      total_drift_abs: 0,
      buy_eur: 0,
      sell_eur: 0,
    },
    diagnostics: {
      reconciliation_ok: true,
      reconciliation_delta_pct: 0,
      unmapped_count: 0,
      fx_missing_count: 0,
      target_not_on_broker_count: 0,
      base_currency: 'EUR',
      sum_eur: 0,
      invested_eur: 0,
      delta_eur: 0,
      tolerance_pct: 0.015,
      stale_price_count: 0,
      entries: [],
    },
    request_id: 1,
  };
}

describe('BuilderStore → BuilderDriftService: global portfolio name propagation', () => {
  let store: BuilderStore;
  let service: BuilderDriftService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BuilderStore,
        BuilderDriftService,
        {
          provide: PipelineBuilderApiService,
          useValue: { runStep: () => EMPTY, getArtifactUrl: () => '' },
        },
      ],
    });
    store = TestBed.inject(BuilderStore);
    service = TestBed.inject(BuilderDriftService);
    http = TestBed.inject(HttpTestingController);
    service.init();
  });

  afterEach(() => http.verify());

  // -------------------------------------------------------------------------
  // Name is embedded in the URL
  // -------------------------------------------------------------------------

  it('when portfolioName is "alpha-fund" and resultStatus is ok, drift URL contains "alpha-fund"', () => {
    store.setPortfolioName('alpha-fund');
    store.setResultStatus('ok');

    service.runExplicit();

    const req = http.expectOne((r) => r.url.includes('alpha-fund'));
    expect(req.request.url).toContain('alpha-fund');
    req.flush(minimalDriftBody());
  });

  it('when portfolioName is "t212" and resultStatus is ok, drift URL contains "t212"', () => {
    store.setPortfolioName('t212');
    store.setResultStatus('ok');

    service.runExplicit();

    const req = http.expectOne((r) => r.url.includes('t212'));
    expect(req.request.url).toContain('t212');
    req.flush(minimalDriftBody());
  });

  // -------------------------------------------------------------------------
  // No UUID in URL when a human-readable name is stored
  // -------------------------------------------------------------------------

  it('when portfolioName is a human-readable name, no UUID segment appears in the drift URL', () => {
    store.setPortfolioName('growth-portfolio');
    store.setResultStatus('ok');

    service.runExplicit();

    const req = http.expectOne((r) => r.url.includes('/drift'));
    expect(req.request.url).not.toMatch(UUID_RE);
    req.flush(minimalDriftBody());
  });

  // -------------------------------------------------------------------------
  // Null guard: no HTTP when store has no name
  // -------------------------------------------------------------------------

  it('when portfolioName is null, no drift HTTP request is fired even with resultStatus ok', () => {
    store.setPortfolioName(null);
    store.setResultStatus('ok');

    service.runExplicit();

    const driftRequests = http.match((r) => r.url.includes('/drift'));
    expect(driftRequests.length).toBe(0);
  });

  // -------------------------------------------------------------------------
  // Name change: subsequent call uses the updated name
  // -------------------------------------------------------------------------

  it('when portfolioName changes from "old" to "new", the subsequent HTTP call uses "new" not "old"', () => {
    store.setPortfolioName('old-portfolio');
    store.setResultStatus('ok');

    // First fetch with "old-portfolio"
    service.runExplicit();
    const first = http.expectOne((r) => r.url.includes('old-portfolio'));
    first.flush(minimalDriftBody());

    // Change the stored name
    store.setPortfolioName('new-portfolio');

    // Second fetch must use the updated name
    service.runExplicit();
    const second = http.expectOne((r) => r.url.includes('new-portfolio'));
    expect(second.request.url).not.toContain('old-portfolio');
    second.flush(minimalDriftBody());
  });

  // -------------------------------------------------------------------------
  // Drift URL shape: contains /portfolio/<name>/drift
  // -------------------------------------------------------------------------

  it('when portfolioName is "core", drift URL matches the /portfolio/core/drift path pattern', () => {
    store.setPortfolioName('core');
    store.setResultStatus('ok');

    service.runExplicit();

    const req = http.expectOne((r) => r.url.includes('/portfolio/core/drift'));
    expect(req.request.method).toBe('GET');
    req.flush(minimalDriftBody());
  });
});
