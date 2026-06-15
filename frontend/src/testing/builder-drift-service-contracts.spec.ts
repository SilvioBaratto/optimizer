/**
 * Source-blind contract tests for BuilderDriftService (issue #1029).
 *
 * Criterion [UNIT]: BuilderDriftService has a request + response contract spec
 * (method/url/body + DriftResponse field-parity).
 *
 * From requirements §10 (portfolio-builder page):
 *   - BuilderStore propagates the globally-selected portfolio name into every
 *     portfolio-name-dependent endpoint call.
 *   - The drift service endpoint uses the correct portfolio name.
 *
 * Request contract: GET portfolio/{encodedName}/drift?base=<currency>
 * Response contract: DriftResponse (holdings, target, drift, trades, totals,
 *   diagnostics, request_id) — the "rich" DriftResponse from drift.model.ts,
 *   distinct from the dashboard simple DriftResponse (entries[]).
 *
 * Assumption: BuilderDriftService fires an HTTP GET when BuilderStore carries a
 * non-null portfolioName and resultStatus is 'ok'. The trigger sequence mirrors
 * portfolio-builder-field-parity.spec.ts:
 *   store.setPortfolioName(name) → store.setResultStatus('ok') →
 *   service.init() → service.runExplicit().
 */

import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

import { BuilderDriftService } from '../app/portfolio-builder/builder-drift.service';
import {
  BuilderStore,
  BUILDER_DRIFT_SERVICE,
} from '../app/portfolio-builder/state/builder.store';
import { schemaOf } from './contract-parity';
import { assertFieldParity } from './contract-field-parity';
import { assertMethod, assertUrl } from './contract-request.helper';
import { makeDriftResponseRich } from './domain-fixtures';
import portfolioSnapshot from './contract-snapshots/portfolio.json';
import { environment } from '../environments/environment';

const API = environment.apiUrl;

// Portfolio name with space to verify percent-encoding.
const PORTFOLIO_NAME = 'tech fund';
const PORTFOLIO_NAME_ENC = encodeURIComponent(PORTFOLIO_NAME);
const DRIFT_URL = `${API}portfolio/${PORTFOLIO_NAME_ENC}/drift`;

function setupDriftService(): {
  service: BuilderDriftService;
  store: BuilderStore;
  http: HttpTestingController;
} {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      BuilderStore,
      BuilderDriftService,
      { provide: BUILDER_DRIFT_SERVICE, useValue: { runExplicit: () => {} } },
    ],
  });
  return {
    service: TestBed.inject(BuilderDriftService),
    store: TestBed.inject(BuilderStore),
    http: TestBed.inject(HttpTestingController),
  };
}

function triggerDrift(
  store: BuilderStore,
  service: BuilderDriftService,
  name: string,
): void {
  store.setPortfolioName(name);
  store.setResultStatus('ok');
  service.init();
  service.runExplicit();
}

describe('BuilderDriftService — request/response contract (issue #1029)', () => {
  let service: BuilderDriftService;
  let store: BuilderStore;
  let http: HttpTestingController;

  beforeEach(() => {
    ({ service, store, http } = setupDriftService());
  });

  afterEach(() => http.verify());

  // ── HTTP method ─────────────────────────────────────────────────────────────

  describe('HTTP method', () => {
    it('when drift is triggered, sends a GET request (not POST/PUT)', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      assertMethod(req, 'GET');
      req.flush(makeDriftResponseRich());
    });

    it('request body is null (GET carries no body)', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.body).toBeNull();
      req.flush(makeDriftResponseRich());
    });
  });

  // ── URL — portfolio name interpolation ──────────────────────────────────────

  describe('URL — portfolio name interpolation', () => {
    it('sends GET to the exact percent-encoded portfolio drift URL', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url === DRIFT_URL);
      assertMethod(req, 'GET');
      assertUrl(req, DRIFT_URL);
      req.flush(makeDriftResponseRich());
    });

    it('portfolio name with spaces is percent-encoded in the URL path', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.url).toContain(PORTFOLIO_NAME_ENC);
      req.flush(makeDriftResponseRich());
    });

    it('no raw {name} token survives in the URL', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.url).not.toContain('{name}');
      req.flush(makeDriftResponseRich());
    });

    it('no raw {id} token survives in the URL', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.url).not.toContain('{id}');
      req.flush(makeDriftResponseRich());
    });

    it('URL contains /portfolio/ path segment followed by /drift', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.url).toContain('portfolio');
      expect(req.request.url).toContain('drift');
      req.flush(makeDriftResponseRich());
    });

    it('when portfolioName changes, the new name is interpolated into the URL', () => {
      triggerDrift(store, service, 'Growth Fund');
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.url).toContain('Growth%20Fund');
      req.flush(makeDriftResponseRich());
    });
  });

  // ── URL — query parameters ──────────────────────────────────────────────────

  describe('URL — query parameters', () => {
    it('appends a `base` query param (not embedded in the URL path)', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.params.has('base')).toBe(true);
      req.flush(makeDriftResponseRich());
    });

    it('`base` query param value is non-empty', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.params.get('base'))
        .withContext('`base` query param must carry a non-empty currency code')
        .toBeTruthy();
      req.flush(makeDriftResponseRich());
    });

    it('`base` param is placed in query params, not in the request body', () => {
      triggerDrift(store, service, PORTFOLIO_NAME);
      const req = http.expectOne((r) => r.url.includes('/drift'));
      expect(req.request.body).toBeNull();
      expect(req.request.params.has('base')).toBe(true);
      req.flush(makeDriftResponseRich());
    });
  });

  // ── Null-portfolio guard ────────────────────────────────────────────────────

  describe('null-portfolio guard', () => {
    it('when portfolioName is null, no HTTP request is sent', () => {
      store.setPortfolioName(null);
      store.setResultStatus('ok');
      service.init();
      service.runExplicit();

      expect(http.match(() => true).length).toBe(0);
    });
  });

  // ── DriftResponse field-parity (portfolio.json schema) ─────────────────────

  describe('DriftResponse — field-parity', () => {
    const schema = schemaOf(portfolioSnapshot, 'DriftResponse');

    it('DriftResponse schema declares the `holdings` property', () => {
      expect('holdings' in (schema.properties ?? {})).toBe(true);
    });

    it('DriftResponse schema declares the `target` property', () => {
      expect('target' in (schema.properties ?? {})).toBe(true);
    });

    it('DriftResponse schema declares the `drift` property', () => {
      expect('drift' in (schema.properties ?? {})).toBe(true);
    });

    it('DriftResponse schema declares the `trades` property', () => {
      expect('trades' in (schema.properties ?? {})).toBe(true);
    });

    it('DriftResponse schema declares the `totals` property', () => {
      expect('totals' in (schema.properties ?? {})).toBe(true);
    });

    it('DriftResponse schema declares the `diagnostics` property', () => {
      expect('diagnostics' in (schema.properties ?? {})).toBe(true);
    });

    it('DriftResponse schema declares `request_id` in snake_case (not camelCase requestId)', () => {
      const props = schema.properties ?? {};
      expect('request_id' in props).toBe(true);
      expect('requestId' in props).toBe(false);
    });

    it('DriftResponseRich fixture has exactly the wire property keys (no missing, no extra)', () => {
      // portfolio.json nests $defs inside DriftResponse, not at the file root,
      // so the ref-resolution root must be the sub-schema, not the snapshot file.
      expect(() =>
        assertFieldParity(schema, makeDriftResponseRich(), schema),
      ).not.toThrow();
    });
  });
});
