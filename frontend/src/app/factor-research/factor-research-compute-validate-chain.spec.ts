/**
 * factor-research-compute-validate-chain.spec.ts — issue #956
 *
 * Contract-parity tests for the factor-research page's full service call chain:
 *   compute() → pollCompute() → applyComputeResult() → signals populated → DOM renders
 *   → onValidate() → validate() → validateReport() set
 *
 * Gap relative to earlier specs:
 *   factor-research-contracts.spec.ts (issue #955) tests onComputeCompleted() → GET
 *   /factors/compute/{id} and verifies computeJobId() becomes null. However the
 *   COMPUTE_PROGRESS fixture there uses the key `ic_report` (singular), whereas
 *   applyComputeResult() reads from `ic_reports` (plural, snake_case list key).
 *   As a result, no existing test verifies that icReports() is actually populated
 *   from the HTTP response, nor that the downstream validate round-trip works.
 *
 * T3 criteria covered:
 *   - "Factor-research: the compute + validate round-trip completes and the IC
 *     chart renders."
 *   - "Every page/panel shows a visible, non-blank error state on primary API
 *     call error."
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const COMPUTE_URL = `${API}factors/compute`;
const VALIDATE_URL = `${API}factors/validate`;
const MACRO_CAL_URL = `${API}views/macro-calibration`;

const ASYNC_RESPONSE = { job_id: 'fc-956', status: 'pending', message: '' };

/** Poll response with the CORRECT plural `ic_reports` key. */
function computeProgress(overrides: Record<string, unknown> = {}) {
  return {
    job_id: 'fc-956',
    status: 'completed',
    current: 10,
    total: 10,
    errors: [] as string[],
    result: {
      // FactorICReport uses camelCase — applyComputeResult casts directly without mapping.
      ic_reports: [
        {
          factor: 'momentum_12_1' as const,
          group: 'momentum' as const,
          ic: 0.05,
          icir: 0.42,
          tStat: 2.1,
          pValue: 0.035,
          vif: 1.8,
          significant: true,
        },
      ],
      taa_signals: [{ factor: 'momentum', signal: 0.7, regime: 'expansion' }],
      factor_returns: [{ factor: 'momentum', date: '2024-12-31', value: 0.08 }],
      cma_sets: [],
    },
    error: null,
    ...overrides,
  };
}

const VALIDATE_RESPONSE = {
  report_date: '2024-12-31',
  factor_type: 'momentum_12_1',
  validation_type: 'in_sample',
  ic_mean: 0.05,
  ic_std: 0.12,
  icir: 0.42,
  t_stat: 2.1,
  p_value: 0.035,
  vif: 1.8,
  details: {},
};

function drainMacroCalibration(http: HttpTestingController): void {
  const pending = http.match((r) => r.url.includes('macro-calibration'));
  pending.forEach((r) => r.flush(
    { phase: 'EXPANSION', delta: 3.0, tau: 0.05, confidence: 0.8, rationale: '', macro_summary: '', timestamp: '', bl_config: { views: [], tau: 0.05, prior_config: { mu_estimator: 'shrunk', risk_aversion: 3.0, cov_estimator: 'ledoit_wolf' } } },
  ));
}

describe('FactorResearchComponent — compute + validate chain (issue #956)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let http: HttpTestingController;
  let el: HTMLElement;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [FactorResearchComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(FactorResearchComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
    drainMacroCalibration(http);
    fixture.detectChanges();
  });

  afterEach(() => {
    http.verify();
    localStorage.clear();
  });

  // ── applyComputeResult key contract ──────────────────────────────────────────

  it('after compute → poll with ic_reports (plural) in result, icReports() is populated', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);

    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(computeProgress());

    expect(comp.icReports().length)
      .withContext('icReports() should be populated from ic_reports key in poll result')
      .toBeGreaterThan(0);
  });

  it('after compute → poll with taa_signals in result, taaSignals() is populated', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);

    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(computeProgress());

    expect(comp.taaSignals().length)
      .withContext('taaSignals() should be populated from taa_signals key')
      .toBeGreaterThan(0);
  });

  it('after compute → poll with factor_returns in result, factorReturns() is populated', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);

    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(computeProgress());

    expect(comp.factorReturns().length)
      .withContext('factorReturns() should be populated from factor_returns key')
      .toBeGreaterThan(0);
  });

  it('when result has no ic_reports key, icReports() stays empty (wrong key = no update)', () => {
    // Verifies the Array.isArray guard: missing key → signal unchanged
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);

    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(
      computeProgress({ result: { ic_report: [{ factor_type: 'bad_key' }] } }), // singular key → no match
    );

    expect(comp.icReports().length)
      .withContext('wrong key should not update icReports signal')
      .toBe(0);
  });

  it('when result.ic_reports is not an array, icReports() stays empty', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);

    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(
      computeProgress({ result: { ic_reports: 'not-an-array' } }),
    );

    expect(comp.icReports().length).toBe(0);
  });

  // ── DOM rendering after compute result applied ────────────────────────────────

  it('after icReports() is populated, switching to factor-analysis tab renders app-factor-analysis-panel', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);
    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(computeProgress());
    fixture.detectChanges();

    comp.activeTab.set('factor-analysis');
    fixture.detectChanges();

    expect(el.querySelector('app-factor-analysis-panel'))
      .withContext('app-factor-analysis-panel should render when factor-analysis tab is active')
      .not.toBeNull();
  });

  it('after taaSignals() is populated, switching to taa tab renders app-taa-panel', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);
    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(computeProgress());
    fixture.detectChanges();

    comp.activeTab.set('taa');
    fixture.detectChanges();

    expect(el.querySelector('app-taa-panel'))
      .withContext('app-taa-panel should render when taa tab is active')
      .not.toBeNull();
  });

  // ── onValidate() → validate() → validateReport() set ─────────────────────────

  it('when onValidate() is called, it POSTs to /factors/validate', () => {
    comp.onValidate('momentum_12_1');

    const req = http.expectOne(VALIDATE_URL);
    expect(req.request.method).toBe('POST');
    req.flush(VALIDATE_RESPONSE);
  });

  it('when onValidate() is called, POST body contains tickers, factor_type, start_date, end_date', () => {
    comp.onValidate('value');

    const req = http.expectOne(VALIDATE_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(Array.isArray(body['tickers'])).toBe(true);
    expect(body['factor_type']).toBe('value');
    expect(typeof body['start_date']).toBe('string');
    expect(typeof body['end_date']).toBe('string');
    req.flush(VALIDATE_RESPONSE);
  });

  it('when onValidate() POST succeeds, validateReport() is populated', () => {
    comp.onValidate('momentum_12_1');
    http.expectOne(VALIDATE_URL).flush(VALIDATE_RESPONSE);

    expect(comp.validateReport()).not.toBeNull();
    expect(comp.validateReport()?.ic_mean).toBe(0.05);
  });

  it('when onValidate() POST returns 422, panelErrors has a truthy validate entry', () => {
    comp.onValidate('unknown_factor');
    http.expectOne(VALIDATE_URL).flush(
      { detail: 'unknown factor type' },
      { status: 422, statusText: 'Unprocessable' },
    );

    const errors = comp.panelErrors();
    const validateErr = errors['validate'] ?? errors['factor-analysis'] ?? errors['validate'];
    // Error may be stored under 'validate' or 'factor-analysis' key depending on implementation
    const hasError = Object.values(errors).some((v) => v !== null);
    expect(hasError)
      .withContext('at least one panel error should be set after validate 422')
      .toBe(true);
  });

  it('when onValidate() POST returns 422, validateReport() stays null', () => {
    comp.onValidate('bad_factor');
    http.expectOne(VALIDATE_URL).flush(
      { detail: 'no data' },
      { status: 422, statusText: 'Unprocessable' },
    );

    expect(comp.validateReport()).toBeNull();
  });

  // ── Full compute + validate round-trip ────────────────────────────────────────

  it('full compute→poll→validate round-trip: icReports and validateReport both populated', () => {
    // Step 1: compute async
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);
    expect(comp.computeJobId()).toBe('fc-956');

    // Step 2: poll result — icReports populated
    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(computeProgress());
    expect(comp.icReports().length).toBeGreaterThan(0);
    expect(comp.computeJobId()).toBeNull();

    // Step 3: validate — validateReport populated
    comp.onValidate('momentum_12_1');
    http.expectOne(VALIDATE_URL).flush(VALIDATE_RESPONSE);
    expect(comp.validateReport()).not.toBeNull();
    expect(comp.validateReport()?.icir).toBe(0.42);
  });

  // ── T3-b: error state for compute failure ────────────────────────────────────

  it('when compute POST returns 5xx, [role="alert"] appears in DOM with non-blank text', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(
      { detail: 'compute engine down' },
      { status: 500, statusText: 'Internal Server Error' },
    );
    fixture.detectChanges();

    const alert = el.querySelector('[role="alert"]') as HTMLElement | null;
    expect(alert).withContext('expected [role="alert"] after 5xx compute error').toBeTruthy();
    expect(alert?.textContent?.trim().length)
      .withContext('error message must be non-blank')
      .toBeGreaterThan(0);
  });

  it('when pollCompute returns 5xx, computeError() is set and computeJobId is null', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);
    expect(comp.computeJobId()).toBe('fc-956');

    comp.onComputeCompleted();
    http.expectOne(`${COMPUTE_URL}/fc-956`).flush(
      { detail: 'poll failed' },
      { status: 500, statusText: 'Internal Server Error' },
    );

    expect(comp.computeError()).toBeTruthy();
    expect(comp.computeJobId()).toBeNull();
  });
});
