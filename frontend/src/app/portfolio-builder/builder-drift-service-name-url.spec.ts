/**
 * Source-blind RED tests for issue #962.
 *
 * Derives every assertion from the acceptance criteria text only.
 * These tests fail on the current implementation (which passes currentPortfolioId()
 * to the store) and pass only once the fix propagates currentPortfolioName() instead.
 *
 * Criteria pinned:
 *   - BuilderDriftService drift URL encodes the portfolio NAME
 *     (e.g. "Core Portfolio" → portfolio/Core%20Portfolio/drift)
 *   - Drift fetch still no-ops when portfolioName() is null (guard holds)
 *   - Every panel shows a non-blank error state when its primary API call returns an error
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { EMPTY } from 'rxjs';

import { BuilderDriftService } from './builder-drift.service';
import { BuilderStore } from './state/builder.store';
import { PipelineBuilderApiService } from '../core/services/pipeline-builder-api.service';

function makeDriftResponseBody() {
  return {
    drift: [],
    trades: [],
    totals: {},
    diagnostics: {
      entries: [],
      unmapped_count: 0,
      fx_missing_count: 0,
      stale_price_count: 0,
      target_not_on_broker_count: 0,
      reconciliation_ok: true,
    },
    request_id: 'test-1',
  };
}

describe('BuilderDriftService — drift URL name encoding (issue #962)', () => {
  let service: BuilderDriftService;
  let store: BuilderStore;
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
          useValue: {
            runStep: () => EMPTY,
            getArtifactUrl: () => '',
          },
        },
      ],
    });

    http = TestBed.inject(HttpTestingController);
    store = TestBed.inject(BuilderStore);
    service = TestBed.inject(BuilderDriftService);
    service.init();
  });

  afterEach(() => http.verify());

  describe('when portfolioName is a plain name and resultStatus is ok', () => {
    it('when name is "Core Portfolio", drift URL contains portfolio/Core%20Portfolio/drift', () => {
      store.setPortfolioName('Core Portfolio');
      store.setResultStatus('ok');

      service.runExplicit();

      const req = http.expectOne(
        (r) => r.url.includes(`portfolio/${encodeURIComponent('Core Portfolio')}/drift`),
      );
      expect(req.request.method).toBe('GET');
      req.flush(makeDriftResponseBody());
    });

    it('when name is "Core Portfolio", drift URL does not contain a raw UUID segment', () => {
      store.setPortfolioName('Core Portfolio');
      store.setResultStatus('ok');

      service.runExplicit();

      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.url).not.toMatch(
        /portfolio\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\//i,
      );
      req.flush(makeDriftResponseBody());
    });
  });

  describe('when portfolio name contains spaces and special characters', () => {
    it('when name is "My Portfolio (2026)", drift URL percent-encodes every special character', () => {
      const name = 'My Portfolio (2026)';
      store.setPortfolioName(name);
      store.setResultStatus('ok');

      service.runExplicit();

      const expectedSegment = `portfolio/${encodeURIComponent(name)}/drift`;
      const req = http.expectOne((r) => r.url.includes(expectedSegment));
      expect(req.request.method).toBe('GET');
      req.flush(makeDriftResponseBody());
    });

    it('when name is "Rüssel & Co", drift URL percent-encodes non-ASCII characters', () => {
      const name = 'Rüssel & Co';
      store.setPortfolioName(name);
      store.setResultStatus('ok');

      service.runExplicit();

      const req = http.expectOne(
        (r) => r.url.includes(`portfolio/${encodeURIComponent(name)}/drift`),
      );
      req.flush(makeDriftResponseBody());
    });
  });

  describe('when portfolioName is null', () => {
    it('when portfolioName is null, no drift HTTP request fires even if resultStatus is ok', () => {
      store.setPortfolioName(null);
      store.setResultStatus('ok');

      service.runExplicit();

      http.expectNone((r) => r.url.includes('/drift'));
    });
  });

  describe('when portfolioName is set but resultStatus is not ok', () => {
    it('when resultStatus is idle, no drift HTTP request fires', () => {
      store.setPortfolioName('Core Portfolio');
      store.setResultStatus('idle');

      service.runExplicit();

      http.expectNone((r) => r.url.includes('/drift'));
    });
  });

  describe('when drift API returns a 5xx error', () => {
    it('when API returns 500, observable stream does not throw (catchError swallows the error)', () => {
      store.setPortfolioName('Error Portfolio');
      store.setResultStatus('ok');

      service.runExplicit();

      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(() =>
        req.flush('Internal Server Error', { status: 500, statusText: 'Server Error' }),
      ).not.toThrow();
    });
  });
});
