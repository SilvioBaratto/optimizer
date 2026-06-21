/**
 * Issue #1048 – DiversificationPreviewComponent contracts (updated for real services).
 *
 * Originally written source-blind; updated to import the real implementation
 * once the component was created at:
 *   frontend/src/app/portfolio-builder/diversification-preview/diversification-preview.ts
 *
 * Tests are re-expressed in terms of the real service APIs:
 *   - YfinanceService.getProfile(id) (called once per ticker)
 *   - MacroIntelligenceService.getMacroCalibration() → MacroCalibrationResponse | null
 *
 * Regime escalation class: flag-escalated (added when phase === 'CONTRACTION').
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection } from '@angular/core';
import { By } from '@angular/platform-browser';
import { BehaviorSubject, of } from 'rxjs';

import { DiversificationPreviewComponent } from '../app/portfolio-builder/diversification-preview/diversification-preview';
import { YfinanceService } from '../app/core/services/yfinance.service';
import { MacroIntelligenceService } from '../app/macro-intelligence/macro-intelligence.service';
import type { TickerProfile } from '../app/core/models/yfinance.model';
import type { MacroCalibrationResponse } from '../app/core/models/macro-intelligence.model';

// ── Fixtures ─────────────────────────────────────────────────────────────────

function stub(sector: string, country: string): TickerProfile {
  return {
    id: 'x', instrument_id: 'x', symbol: null, short_name: null, long_name: null,
    isin: null, exchange: null, quote_type: null, currency: null,
    sector, industry: null, country,
    website: null, long_business_summary: null, market_cap: null,
    enterprise_value: null, shares_outstanding: null, float_shares: null,
    current_price: null, previous_close: null, fifty_two_week_low: null,
    fifty_two_week_high: null, fifty_day_average: null, two_hundred_day_average: null,
    average_volume: null, beta: null, trailing_pe: null, forward_pe: null,
    trailing_eps: null, forward_eps: null, price_to_sales_trailing_12months: null,
    price_to_book: null, enterprise_to_revenue: null, enterprise_to_ebitda: null,
    peg_ratio: null, book_value: null, profit_margins: null, operating_margins: null,
    gross_margins: null, return_on_assets: null, return_on_equity: null,
    total_revenue: null, revenue_growth: null, earnings_growth: null, ebitda: null,
    free_cashflow: null, operating_cashflow: null, total_debt: null,
    debt_to_equity: null, current_ratio: null, dividend_rate: null,
    dividend_yield: null, payout_ratio: null, recommendation_key: null,
    recommendation_mean: null, full_time_employees: null,
    created_at: '', updated_at: '',
  };
}

function calibration(phase: string): MacroCalibrationResponse {
  return {
    phase: phase as MacroCalibrationResponse['phase'],
    delta: 0, tau: 0, confidence: 0.8,
    rationale: '', macro_summary: '', timestamp: '',
    bl_config: { views: [], tau: 0, prior_config: { mu_estimator: '', risk_aversion: 0, cov_estimator: '' } },
  };
}

// All three assets in United States → region 'North America' = 100 % > 60 %
const ALL_US = [
  stub('Technology', 'United States'),
  stub('Technology', 'United States'),
  stub('Consumer',   'United States'),
];

// ── Suite ─────────────────────────────────────────────────────────────────────

describe('DiversificationPreviewComponent (issue-1048 — real-service wiring)', () => {
  let fixture: ComponentFixture<DiversificationPreviewComponent>;
  let yfinanceSpy: jasmine.SpyObj<YfinanceService>;
  let macroSpy: jasmine.SpyObj<MacroIntelligenceService>;
  let macroSubject: BehaviorSubject<MacroCalibrationResponse | null>;

  beforeEach(async () => {
    yfinanceSpy = jasmine.createSpyObj<YfinanceService>('YfinanceService', ['getProfile']);
    macroSpy    = jasmine.createSpyObj<MacroIntelligenceService>(
      'MacroIntelligenceService', ['getMacroCalibration'],
    );

    yfinanceSpy.getProfile.and.returnValue(of(stub('Technology', 'United States')));
    macroSubject = new BehaviorSubject<MacroCalibrationResponse | null>(calibration('EARLY_EXPANSION'));
    macroSpy.getMacroCalibration.and.returnValue(macroSubject.asObservable());

    await TestBed.configureTestingModule({
      imports: [DiversificationPreviewComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        { provide: YfinanceService,           useValue: yfinanceSpy },
        { provide: MacroIntelligenceService,  useValue: macroSpy    },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(DiversificationPreviewComponent);
    fixture.detectChanges();
  });

  // ======================================================================
  // Criterion A – fetch + chart render
  // ======================================================================

  describe('Criterion A — asset selection triggers profile + regime fetch and chart render', () => {

    it('when tickers are set, fetches TickerProfiles for those tickers', () => {
      fixture.componentRef.setInput('tickers', ['aapl', 'rog']);
      fixture.detectChanges();

      expect(yfinanceSpy.getProfile).toHaveBeenCalledWith('aapl');
      expect(yfinanceSpy.getProfile).toHaveBeenCalledWith('rog');
    });

    it('when tickers are set, fetches the current macro regime', () => {
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      expect(macroSpy.getMacroCalibration).toHaveBeenCalled();
    });

    it('when profiles load, renders a sector breakdown chart', () => {
      fixture.componentRef.setInput('tickers', ['aapl', 'rog']);
      fixture.detectChanges();

      const chart =
        fixture.debugElement.query(By.css('app-echarts-donut[data-chart="sector"]')) ??
        fixture.debugElement.query(By.css('[data-chart="sector"]'));

      expect(chart).withContext('sector chart element must be present in DOM').not.toBeNull();
    });

    it('when profiles load, renders a region breakdown chart', () => {
      fixture.componentRef.setInput('tickers', ['aapl', 'rog']);
      fixture.detectChanges();

      const chart =
        fixture.debugElement.query(By.css('[data-chart="region"]'));

      expect(chart).withContext('region chart element must be present in DOM').not.toBeNull();
    });

    it('when tickers change from one set to another, re-fetches profiles for the new tickers', () => {
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      fixture.componentRef.setInput('tickers', ['msft', 'rog']);
      fixture.detectChanges();

      expect(yfinanceSpy.getProfile).toHaveBeenCalledWith('msft');
      expect(yfinanceSpy.getProfile).toHaveBeenCalledWith('rog');
    });

  });

  // ======================================================================
  // Criterion B – Regime-aware flag badges
  // ======================================================================

  describe('Criterion B — regime-aware flag badges per violation', () => {

    it('when a single region exceeds 60 %, renders a region-concentration flag badge', () => {
      yfinanceSpy.getProfile.and.returnValues(...ALL_US.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['aapl', 'msft', 'amzn']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="region-concentration"]')))
        .withContext('region-concentration badge must be visible')
        .not.toBeNull();
    });

    it('when a sector exceeds 15 %, renders a sector-concentration flag badge', () => {
      const profiles = [
        stub('Technology', 'United States'),
        stub('Technology', 'United States'),
        stub('Technology', 'United States'),
        stub('Healthcare', 'Germany'),
      ];
      yfinanceSpy.getProfile.and.returnValues(...profiles.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['t1', 't2', 't3', 't4']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="sector-concentration"]')))
        .not.toBeNull();
    });

    it('when HHI >= 0.12, renders an HHI flag badge', () => {
      yfinanceSpy.getProfile.and.returnValues(
        of(stub('Technology', 'United States')),
        of(stub('Technology', 'United States')),
      );
      fixture.componentRef.setInput('tickers', ['t1', 't2']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="hhi"]'))).not.toBeNull();
    });

    it('when top-4 holdings >= 30 %, renders a top4-concentration flag badge', () => {
      yfinanceSpy.getProfile.and.returnValues(
        of(stub('Technology', 'United States')),
        of(stub('Technology', 'United States')),
      );
      fixture.componentRef.setInput('tickers', ['t1', 't2']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="top4-concentration"]'))).not.toBeNull();
    });

    it('when healthcare weight is below 8 %, renders a healthcare flag badge', () => {
      yfinanceSpy.getProfile.and.returnValue(of(stub('Technology', 'United States')));
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="healthcare"]'))).not.toBeNull();
    });

    it('when technology weight is below 10 %, renders a technology flag badge', () => {
      yfinanceSpy.getProfile.and.returnValue(of(stub('Healthcare', 'Germany')));
      fixture.componentRef.setInput('tickers', ['rog']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="technology"]'))).not.toBeNull();
    });

    it('when a major sector is entirely absent, renders an absent-sector flag badge', () => {
      yfinanceSpy.getProfile.and.returnValue(of(stub('Technology', 'United States')));
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="absent-sector"]'))).not.toBeNull();
    });

    // ── Regime escalation ────────────────────────────────────────────────

    it('when regime is CONTRACTION and a violation exists, flag badge carries flag-escalated class', () => {
      macroSubject.next(calibration('CONTRACTION'));
      yfinanceSpy.getProfile.and.returnValues(...ALL_US.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['aapl', 'msft', 'amzn']);
      fixture.detectChanges();

      const badge = fixture.debugElement.query(By.css('[data-flag="region-concentration"]'));
      expect(badge).not.toBeNull();
      expect(badge.nativeElement.classList)
        .withContext('CONTRACTION → badge must carry flag-escalated class')
        .toContain('flag-escalated');
    });

    it('when regime is EARLY_EXPANSION and a violation exists, badge does NOT carry flag-escalated class', () => {
      macroSubject.next(calibration('EARLY_EXPANSION'));
      yfinanceSpy.getProfile.and.returnValues(...ALL_US.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['aapl', 'msft', 'amzn']);
      fixture.detectChanges();

      const badge = fixture.debugElement.query(By.css('[data-flag="region-concentration"]'));
      expect(badge).not.toBeNull();
      expect(badge.nativeElement.classList)
        .withContext('expansion regime → badge must NOT carry flag-escalated class')
        .not.toContain('flag-escalated');
    });

    it('when multiple violations coexist, each renders a separate flag badge', () => {
      yfinanceSpy.getProfile.and.returnValues(
        of(stub('Technology', 'United States')),
        of(stub('Technology', 'United States')),
      );
      fixture.componentRef.setInput('tickers', ['t1', 't2']);
      fixture.detectChanges();

      const badges = fixture.debugElement.queryAll(By.css('[data-flag]'));
      expect(badges.length)
        .withContext('each independent violation must have its own flag badge')
        .toBeGreaterThan(1);
    });

  });
});
