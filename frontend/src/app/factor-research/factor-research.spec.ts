import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of } from 'rxjs';

import { FactorResearchComponent } from './factor-research';
import { FactorsService } from './factors.service';
import { ICON_PROVIDER } from '../icons';
import type {
  FactorExposureConstraintsApiResponse,
  FactorExposureConstraintsRequest,
  FactorScoreApiResponse,
  FactorScoreRequest,
  FactorSelectApiResponse,
  FactorSelectRequest,
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
