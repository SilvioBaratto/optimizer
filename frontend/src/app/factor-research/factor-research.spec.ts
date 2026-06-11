import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of, throwError } from 'rxjs';

import { FactorResearchComponent } from './factor-research';
import { FactorsService } from './factors.service';
import { ICON_PROVIDER } from '../icons';
import type {
  FactorComputeAsyncResponse,
  FactorExposureConstraintsApiResponse,
  FactorExposureConstraintsRequest,
  FactorQuintileSpreadApiResponse,
  FactorScoreApiResponse,
  FactorScoreRequest,
  FactorSelectApiResponse,
  FactorSelectRequest,
  FactorValidateResponse,
} from './factor.model';
import type { MacroCalibrationResponse, BlackLittermanBlConfig } from '../core/models/macro-intelligence.model';

function scoreResponse(): FactorScoreApiResponse {
  return {
    score_date: '2026-04-18',
    scores: { AAPL: 1.2, MSFT: 0.9, GOOG: 0.7 },
    group_contributions: {},
  };
}

describe('FactorResearchComponent — Score tab wiring (#454)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let component: FactorResearchComponent;
  let factors: FactorsService;
  let scoreSpy: jasmine.Spy<(req: FactorScoreRequest) => ReturnType<FactorsService['score']>>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FactorResearchComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    factors = TestBed.inject(FactorsService);
    scoreSpy = spyOn(factors, 'score').and.returnValue(of(scoreResponse()));
    fixture = TestBed.createComponent(FactorResearchComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it("exposes a 'score' tab labelled 'Scoring'", () => {
    const tab = component.tabs.find((t) => t.id === 'score');
    expect(tab).toBeDefined();
    expect(tab!.label).toBe('Scoring');
  });

  it("mounts <app-score-panel> when activeTab is 'score'", () => {
    component.activeTab.set('score');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-score-panel').length).toBe(1);
  });

  it("does NOT mount <app-score-panel> when activeTab is a different tab", () => {
    component.activeTab.set('regime');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-score-panel').length).toBe(0);
  });

  it("forwards a runScore emission to FactorsService.score() with the exact payload", () => {
    const payload: FactorScoreRequest = {
      tickers: ['AAPL', 'MSFT'],
      score_date: '2026-04-18',
      composite_method: 'equal_weight',
    };

    component.onRunScore(payload);

    expect(scoreSpy).toHaveBeenCalledTimes(1);
    expect(scoreSpy).toHaveBeenCalledWith(payload);
  });

  it("stores the score response in scoreResult and clears scoreLoading on success", () => {
    component.onRunScore({
      tickers: ['AAPL'],
      score_date: '2026-04-18',
      composite_method: 'equal_weight',
    });

    expect(component.scoreResult()).toEqual(scoreResponse());
    expect(component.scoreLoading()).toBe(false);
  });
});

describe('FactorResearchComponent — Select tab wiring (#455)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let component: FactorResearchComponent;
  let factors: FactorsService;
  let selectSpy: jasmine.Spy<
    (req: FactorSelectRequest) => ReturnType<FactorsService['select']>
  >;

  function selectResponse(): FactorSelectApiResponse {
    return {
      selected_tickers: ['AAPL', 'MSFT'],
      count: 2,
      turnover: 0.1,
      buffer_zone: { entered: [], exited: [] },
    };
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FactorResearchComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    factors = TestBed.inject(FactorsService);
    selectSpy = spyOn(factors, 'select').and.returnValue(of(selectResponse()));
    fixture = TestBed.createComponent(FactorResearchComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it("exposes a 'select' tab labelled 'Selection'", () => {
    const tab = component.tabs.find((t) => t.id === 'select');
    expect(tab).toBeDefined();
    expect(tab!.label).toBe('Selection');
  });

  it("mounts <app-select-panel> when activeTab is 'select'", () => {
    component.activeTab.set('select');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-select-panel').length).toBe(1);
  });

  it("does NOT mount <app-select-panel> when activeTab is a different tab", () => {
    component.activeTab.set('regime');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-select-panel').length).toBe(0);
  });

  it("forwards a runSelect emission to FactorsService.select() with the exact payload", () => {
    const payload: FactorSelectRequest = {
      tickers: ['AAPL', 'MSFT'],
      start_date: '2025-04-18',
      end_date: '2026-04-18',
      method: 'fixed_count',
      sector_balance: false,
      target_count: 30,
    };

    component.onRunSelect(payload);

    expect(selectSpy).toHaveBeenCalledTimes(1);
    expect(selectSpy).toHaveBeenCalledWith(payload);
  });

  it("stores the select response in selectResult and clears selectLoading on success", () => {
    component.onRunSelect({
      tickers: ['AAPL'],
      start_date: '2025-04-18',
      end_date: '2026-04-18',
      method: 'fixed_count',
      sector_balance: false,
      target_count: 30,
    });

    expect(component.selectResult()).toEqual(selectResponse());
    expect(component.selectLoading()).toBe(false);
  });
});

describe('FactorResearchComponent — Exposure Constraints tab wiring (#456)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let component: FactorResearchComponent;
  let factors: FactorsService;
  let constraintsSpy: jasmine.Spy<
    (req: FactorExposureConstraintsRequest) => ReturnType<FactorsService['exposureConstraints']>
  >;

  function constraintsResponse(): FactorExposureConstraintsApiResponse {
    return {
      left_inequality: [[1, 0], [0, 1]],
      right_inequality: [0.5, 0.5],
    };
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FactorResearchComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    factors = TestBed.inject(FactorsService);
    constraintsSpy = spyOn(factors, 'exposureConstraints').and.returnValue(
      of(constraintsResponse()),
    );
    fixture = TestBed.createComponent(FactorResearchComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it("exposes an 'exposure-constraints' tab labelled 'Exposure Constraints'", () => {
    const tab = component.tabs.find((t) => t.id === 'exposure-constraints');
    expect(tab).toBeDefined();
    expect(tab!.label).toBe('Exposure Constraints');
  });

  it("mounts <app-exposure-constraints-panel> when activeTab is 'exposure-constraints'", () => {
    component.activeTab.set('exposure-constraints');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-exposure-constraints-panel').length).toBe(1);
  });

  it("does NOT mount the panel when activeTab is a different tab", () => {
    component.activeTab.set('regime');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-exposure-constraints-panel').length).toBe(0);
  });

  it("forwards a runConstraints emission to FactorsService.exposureConstraints() with the exact payload", () => {
    const payload: FactorExposureConstraintsRequest = {
      tickers: ['AAPL', 'MSFT'],
      start_date: '2025-04-18',
      end_date: '2026-04-18',
      bounds: { momentum_12_1: [-0.5, 0.5] },
    };

    component.onRunExposureConstraints(payload);

    expect(constraintsSpy).toHaveBeenCalledTimes(1);
    expect(constraintsSpy).toHaveBeenCalledWith(payload);
  });

  it("stores the response in exposureConstraintsResult and clears loading on success", () => {
    component.onRunExposureConstraints({
      tickers: ['AAPL'],
      start_date: '2025-04-18',
      end_date: '2026-04-18',
      bounds: { volatility: [-0.3, 0.3] },
    });

    expect(component.exposureConstraintsResult()).toEqual(constraintsResponse());
    expect(component.exposureConstraintsLoading()).toBe(false);
  });
});

describe('FactorResearchComponent — macro-calibration stat-card computeds (#494)', () => {
  let component: FactorResearchComponent;

  const blConfig: BlackLittermanBlConfig = {
    views: [],
    tau: 0.05,
    prior_config: { mu_estimator: 'shrunk', risk_aversion: 3.5, cov_estimator: 'ledoit_wolf' },
  };
  const cal: MacroCalibrationResponse = {
    phase: 'EARLY_EXPANSION',
    delta: 3.5,
    tau: 0.05,
    confidence: 0.78,
    rationale: 'Leading indicators positive',
    macro_summary: 'PMI above 55',
    timestamp: '2026-04-28T00:00:00Z',
    bl_config: blConfig,
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FactorResearchComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();
    component = TestBed.createComponent(FactorResearchComponent).componentInstance;
  });

  it('macroPhase returns "—" when macroCalibration is null', () => {
    expect(component.macroPhase()).toBe('—');
  });

  it('macroPhase returns the phase string when calibration is set', () => {
    component.macroCalibration.set(cal);
    expect(component.macroPhase()).toBe('EARLY_EXPANSION');
  });

  it('macroConfidence returns "—" when macroCalibration is null', () => {
    expect(component.macroConfidence()).toBe('—');
  });

  it('macroConfidence formats confidence as integer percentage string', () => {
    component.macroCalibration.set(cal);
    expect(component.macroConfidence()).toBe('78%');
  });

  it('macroDelta returns "—" when macroCalibration is null', () => {
    expect(component.macroDelta()).toBe('—');
  });

  it('macroDelta formats delta with two decimal places', () => {
    component.macroCalibration.set(cal);
    expect(component.macroDelta()).toBe('3.50');
  });
});

// ── Method coverage (#941) ───────────────────────────────────────────────────
// Spies cover each handler's success path; a message-less `throwError(() => ({}))`
// covers the error path AND the `?? 'default'` fallback (an HttpErrorResponse
// always has a `.message`, so the default is otherwise unreachable).
describe('FactorResearchComponent — method coverage (#941)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let factors: FactorsService;

  const okScore = { score_date: '2026-01-01', scores: { AAPL: 1 }, group_contributions: {} } as FactorScoreApiResponse;
  const okSelect = { selected_tickers: ['AAPL'], count: 1, turnover: 0, buffer_zone: { entered: [], exited: [] } } as FactorSelectApiResponse;
  const okConstraints = { left_inequality: [[1]], right_inequality: [0.5] } as FactorExposureConstraintsApiResponse;
  const okValidate = {} as FactorValidateResponse;
  const okQuintile = {} as FactorQuintileSpreadApiResponse;
  const okCompute = { job_id: 'j1', status: 'pending', message: '' } as FactorComputeAsyncResponse;
  const okCal = {
    phase: 'EARLY_EXPANSION', delta: 1, tau: 0.1, confidence: 0.5, rationale: 'r',
    macro_summary: 's', timestamp: 't',
    bl_config: { views: [], tau: 0.1, prior_config: { mu_estimator: 'm', risk_aversion: 1, cov_estimator: 'c' } },
  } as MacroCalibrationResponse;
  const boom = () => throwError(() => ({}));

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FactorResearchComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    factors = TestBed.inject(FactorsService);
    spyOn(factors, 'macroCalibration').and.returnValue(of(okCal));
    spyOn(factors, 'compute').and.returnValue(of(okCompute));
    spyOn(factors, 'validate').and.returnValue(of(okValidate));
    spyOn(factors, 'score').and.returnValue(of(okScore));
    spyOn(factors, 'select').and.returnValue(of(okSelect));
    spyOn(factors, 'exposureConstraints').and.returnValue(of(okConstraints));
    spyOn(factors, 'quintileSpread').and.returnValue(of(okQuintile));
    spyOn(factors, 'teObservations').and.returnValue(of([]));

    fixture = TestBed.createComponent(FactorResearchComponent);
    comp = fixture.componentInstance;
    fixture.detectChanges(); // ngOnInit → macroCalibration success
  });

  it('ngOnInit populates macroCalibration on success', () => {
    expect(comp.macroCalibration()).not.toBeNull();
  });

  it('onRunCompute stores the job id on success', () => {
    comp.onRunCompute();
    expect(comp.computeJobId()).toBe('j1');
  });

  it('onRunCompute records the default error on a message-less failure', () => {
    (factors.compute as jasmine.Spy).and.returnValue(boom());
    comp.computeJobId.set(null);
    comp.onRunCompute();
    expect(comp.computeError()).toBe('Compute failed');
  });

  it('onComputeCompleted clears the job id', () => {
    comp.computeJobId.set('j1');
    comp.onComputeCompleted();
    expect(comp.computeJobId()).toBeNull();
  });

  it('onComputeFailed clears the job and sets the error (with default fallback)', () => {
    comp.onComputeFailed('boom');
    expect(comp.computeError()).toBe('boom');
    comp.onComputeFailed('');
    expect(comp.computeError()).toBe('Compute job failed');
  });

  it('onValidate stores the report on success and the default error on failure', () => {
    comp.onValidate('momentum');
    expect(comp.validateReport()).not.toBeNull();
    (factors.validate as jasmine.Spy).and.returnValue(boom());
    comp.onValidate('momentum');
    expect(comp.panelErrors()['validate']).toBe('Validate failed');
  });

  it('onQuintileSpread stores the result and the default error on failure', () => {
    comp.onQuintileSpread('momentum');
    expect(comp.quintileSpread()).not.toBeNull();
    (factors.quintileSpread as jasmine.Spy).and.returnValue(boom());
    comp.onQuintileSpread('momentum');
    expect(comp.panelErrors()['quintile']).toBe('Quintile spread failed');
  });

  it('onRunScore records the default error on a message-less failure', () => {
    (factors.score as jasmine.Spy).and.returnValue(boom());
    comp.onRunScore({ tickers: ['AAPL'], score_date: '2026-01-01', composite_method: 'equal_weight' });
    expect(comp.scoreLoading()).toBe(false);
    expect(comp.panelErrors()['score']).toBe('Score run failed');
  });

  it('onRunSelect records the default error on a message-less failure', () => {
    (factors.select as jasmine.Spy).and.returnValue(boom());
    comp.onRunSelect({ tickers: ['AAPL'], start_date: '2025-01-01', end_date: '2026-01-01', method: 'fixed_count', sector_balance: false, target_count: 10 });
    expect(comp.selectLoading()).toBe(false);
    expect(comp.panelErrors()['select']).toBe('Select run failed');
  });

  it('onRunExposureConstraints records the default error on a message-less failure', () => {
    (factors.exposureConstraints as jasmine.Spy).and.returnValue(boom());
    comp.onRunExposureConstraints({ tickers: ['AAPL'], start_date: '2025-01-01', end_date: '2026-01-01', bounds: {} });
    expect(comp.exposureConstraintsLoading()).toBe(false);
    expect(comp.panelErrors()['exposure-constraints']).toBe('Exposure constraints run failed');
  });

  it('onFetchTe stores observations on success and the default error on failure', () => {
    comp.onFetchTe();
    expect(comp.teObservations()).toEqual([]);
    (factors.teObservations as jasmine.Spy).and.returnValue(boom());
    comp.onFetchTe();
    expect(comp.panelErrors()['te']).toBe('TE fetch failed');
  });

  it('fetchMacroCalibration records the default error on a message-less failure', () => {
    (factors.macroCalibration as jasmine.Spy).and.returnValue(boom());
    comp.macroCalibration.set(null);
    comp.ngOnInit();
    expect(comp.panelErrors()['macro-calibration']).toBe('Macro calibration failed');
  });

  it('retry clears the error state', () => {
    comp.hasError.set(true);
    comp.errorMessage.set('x');
    comp.retry();
    expect(comp.hasError()).toBe(false);
    expect(comp.errorMessage()).toBe('');
  });
});
