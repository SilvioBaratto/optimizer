import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';
import { throwError } from 'rxjs';

import { configureTestBed, injectHttp } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { MomentPanelComponent } from './moment-panel';
import { OptimizationService } from '../optimization.service';
import { environment } from '../../../environments/environment';

const ADAPT_URL = `${environment.apiUrl}llm-moments/adapt-factor-weights`;
const CALIBRATE_URL = `${environment.apiUrl}llm-moments/calibrate-delta`;
const REGIME_URL = `${environment.apiUrl}llm-moments/select-cov-regime`;

describe('MomentPanelComponent', () => {
  let fixture: ComponentFixture<MomentPanelComponent>;
  let comp: MomentPanelComponent;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ imports: [MomentPanelComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(MomentPanelComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('when macro indicators are below 20 chars, no request fires and an error is set', () => {
    comp.macroIndicators.set('a'.repeat(19));
    comp.runAdaptFactorWeights();
    http.expectNone(ADAPT_URL);
    expect(comp.adaptError()).toContain('at least 20');
  });

  it('when macro indicators reach 20 chars, the adapt-factor-weights request fires', () => {
    comp.macroIndicators.set('a'.repeat(20));
    comp.runAdaptFactorWeights();
    const req = http.expectOne(ADAPT_URL);
    expect(req.request.method).toBe('POST');
    req.flush({ phase: 'expansion', weights: { value: 0.5 } });
    expect(comp.factorWeights()).toEqual({ value: 0.5 });
  });

  it('when the request errors, calibrateDeltaError-style error is surfaced', () => {
    comp.macroIndicators.set('a'.repeat(25));
    comp.runAdaptFactorWeights();
    http.expectOne(ADAPT_URL).flush({ detail: 'llm down' }, { status: 502, statusText: 'Bad Gateway' });
    expect(comp.adaptError()).toBeTruthy();
  });

  // ── calibrate-delta (issue #989: AC1 request shape + AC3 render/error) ────────

  it('when macro text is provided, calibrate-delta posts {macro_text}', () => {
    comp.macroText.set('Fed hiked rates, inflation sticky, vol elevated');
    comp.runCalibrateDelta();
    const req = http.expectOne(CALIBRATE_URL);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({
      macro_text: 'Fed hiked rates, inflation sticky, vol elevated',
    });
    req.flush({ delta: 4.2, rationale: 'elevated vol' });
    expect(comp.delta()).toBe(4.2);
  });

  it('when macro text is blank, no calibrate-delta request fires', () => {
    comp.macroText.set('   ');
    comp.runCalibrateDelta();
    expect(http.match(CALIBRATE_URL).length).toBe(0);
  });

  it('when calibrate-delta succeeds, the delta value is rendered', () => {
    comp.macroText.set('some macro regime description');
    comp.runCalibrateDelta();
    http.expectOne(CALIBRATE_URL).flush({ delta: 4.2, rationale: 'r' });
    fixture.detectChanges();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain('4.2');
  });

  it('when calibrate-delta errors, a visible error message is rendered', () => {
    comp.macroText.set('some macro regime description');
    comp.runCalibrateDelta();
    http
      .expectOne(CALIBRATE_URL)
      .flush({ detail: 'llm down' }, { status: 502, statusText: 'Bad Gateway' });
    fixture.detectChanges();
    expect(comp.calibrateDeltaError()).toBeTruthy();
    expect((fixture.nativeElement as HTMLElement).textContent).toContain(
      comp.calibrateDeltaError()!,
    );
  });

  // ── select-cov-regime (issue #989: AC1 request shape + AC3 render/error) ──────

  it('when select-cov-regime runs, it posts the three-field body', () => {
    comp.newsHeadlinesRaw.set('sell-off in tech');
    comp.avgSentiment.set(-0.3);
    comp.realizedVol.set(0.25);
    comp.runSelectCovRegime();
    const req = http.expectOne(REGIME_URL);
    expect(req.request.body).toEqual({
      news_headlines: ['sell-off in tech'],
      avg_sentiment_score: -0.3,
      realized_vol_30d: 0.25,
    });
    req.flush({ estimator_type: 'ledoit_wolf', rationale: 'elevated vol' });
    expect(comp.regimeEstimator()).toBe('ledoit_wolf');
  });

  it('when select-cov-regime errors, a visible error is set', () => {
    comp.newsHeadlinesRaw.set('headline');
    comp.runSelectCovRegime();
    http
      .expectOne(REGIME_URL)
      .flush({ detail: 'llm error' }, { status: 502, statusText: 'Bad Gateway' });
    expect(comp.regimeError()).toBeTruthy();
  });

  // ── Guards & result rendering (branch coverage) ──────────────────────────────

  it('when factor groups are empty, no adapt-factor-weights request fires', () => {
    comp.factorGroupsRaw.set('');
    comp.macroIndicators.set('a'.repeat(25));
    comp.runAdaptFactorWeights();
    expect(http.match(ADAPT_URL).length).toBe(0);
  });

  it('when adapt succeeds, the phase and factor weights render', () => {
    comp.macroIndicators.set('a'.repeat(25));
    comp.runAdaptFactorWeights();
    http.expectOne(ADAPT_URL).flush({ phase: 'expansion', weights: { value: 1.25 } });
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('expansion');
    expect(text).toContain('1.250');
  });

  // ── Message-less failures exercise the `?? default` error fallbacks ───────────

  it('when calibrate-delta fails without a message, the default error is used', () => {
    spyOn(TestBed.inject(OptimizationService), 'calibrateDelta').and.returnValue(
      throwError(() => ({})),
    );
    comp.macroText.set('macro regime text');
    comp.runCalibrateDelta();
    expect(comp.calibrateDeltaError()).toBe('calibrate-delta failed');
  });

  it('when adapt-factor-weights fails without a message, the default error is used', () => {
    spyOn(TestBed.inject(OptimizationService), 'adaptFactorWeights').and.returnValue(
      throwError(() => ({})),
    );
    comp.macroIndicators.set('a'.repeat(25));
    comp.runAdaptFactorWeights();
    expect(comp.adaptError()).toBe('adapt-factor-weights failed');
  });

  it('when select-cov-regime fails without a message, the default error is used', () => {
    spyOn(TestBed.inject(OptimizationService), 'selectCovRegime').and.returnValue(
      throwError(() => ({})),
    );
    comp.newsHeadlinesRaw.set('headline');
    comp.runSelectCovRegime();
    expect(comp.regimeError()).toBe('select-cov-regime failed');
  });
});
