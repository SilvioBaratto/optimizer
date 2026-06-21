/**
 * DiversificationPreviewComponent — R4 acceptance-criteria spec (issue #1048).
 *
 * Criterion A: Selecting assets fetches TickerProfiles + current regime,
 *   then renders sector and region breakdown charts.
 *
 * Criterion B: Regime-aware flag badges render for each violated criterion,
 *   with escalation class when regime is CONTRACTION.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection } from '@angular/core';
import { By } from '@angular/platform-browser';
import { BehaviorSubject, of } from 'rxjs';

import { DiversificationPreviewComponent } from './diversification-preview';
import { YfinanceService } from '../../core/services/yfinance.service';
import { MacroIntelligenceService } from '../../macro-intelligence/macro-intelligence.service';
import type { TickerProfile } from '../../core/models/yfinance.model';
import type { MacroCalibrationResponse } from '../../core/models/macro-intelligence.model';

// ── Fixtures ─────────────────────────────────────────────────────────────────

function makeProfile(sector: string, country: string): TickerProfile {
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

function makeCalibration(phase: string): MacroCalibrationResponse {
  return {
    phase: phase as MacroCalibrationResponse['phase'],
    delta: 0, tau: 0, confidence: 0.8,
    rationale: '', macro_summary: '', timestamp: '',
    bl_config: { views: [], tau: 0, prior_config: { mu_estimator: '', risk_aversion: 0, cov_estimator: '' } },
  };
}

// All assets in US → region 'North America' = 100 % > 60 % → region-concentration
const ALL_US_PROFILES = [
  makeProfile('Technology', 'United States'),
  makeProfile('Technology', 'United States'),
  makeProfile('Consumer', 'United States'),
];

// ── Setup ─────────────────────────────────────────────────────────────────────

describe('DiversificationPreviewComponent (issue-1048)', () => {
  let fixture: ComponentFixture<DiversificationPreviewComponent>;
  let yfinanceSpy: jasmine.SpyObj<YfinanceService>;
  let macroSpy: jasmine.SpyObj<MacroIntelligenceService>;
  let macroSubject: BehaviorSubject<MacroCalibrationResponse | null>;

  beforeEach(async () => {
    yfinanceSpy = jasmine.createSpyObj<YfinanceService>('YfinanceService', ['getProfile']);
    macroSpy = jasmine.createSpyObj<MacroIntelligenceService>(
      'MacroIntelligenceService', ['getMacroCalibration'],
    );

    yfinanceSpy.getProfile.and.returnValue(of(makeProfile('Technology', 'United States')));
    macroSubject = new BehaviorSubject<MacroCalibrationResponse | null>(makeCalibration('EARLY_EXPANSION'));
    macroSpy.getMacroCalibration.and.returnValue(macroSubject.asObservable());

    await TestBed.configureTestingModule({
      imports: [DiversificationPreviewComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        { provide: YfinanceService, useValue: yfinanceSpy },
        { provide: MacroIntelligenceService, useValue: macroSpy },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(DiversificationPreviewComponent);
    fixture.detectChanges();
  });

  // ======================================================================
  // Criterion A — fetch + chart render
  // ======================================================================

  describe('Criterion A — asset selection triggers fetch and chart rendering', () => {

    it('when tickers are set, calls getProfile for each ticker', () => {
      fixture.componentRef.setInput('tickers', ['aapl', 'rog']);
      fixture.detectChanges();

      expect(yfinanceSpy.getProfile).toHaveBeenCalledWith('aapl');
      expect(yfinanceSpy.getProfile).toHaveBeenCalledWith('rog');
    });

    it('when tickers are set, fetches the current macro calibration', () => {
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      expect(macroSpy.getMacroCalibration).toHaveBeenCalled();
    });

    it('when profiles load, renders a sector breakdown chart', () => {
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      const chart =
        fixture.debugElement.query(By.css('app-echarts-donut[data-chart="sector"]')) ??
        fixture.debugElement.query(By.css('[data-chart="sector"]'));

      expect(chart).withContext('sector chart must be in DOM').not.toBeNull();
    });

    it('when profiles load, renders a region breakdown chart', () => {
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      const chart =
        fixture.debugElement.query(By.css('[data-chart="region"]'));

      expect(chart).withContext('region chart must be in DOM').not.toBeNull();
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
  // Criterion B — regime-aware flag badges
  // ======================================================================

  describe('Criterion B — flag badges for violated criteria', () => {

    it('when region concentration > 60 %, renders a region-concentration flag badge', () => {
      yfinanceSpy.getProfile.and.returnValues(
        ...ALL_US_PROFILES.map(p => of(p)),
      );
      fixture.componentRef.setInput('tickers', ['aapl', 'msft', 'amzn']);
      fixture.detectChanges();

      const badge = fixture.debugElement.query(By.css('[data-flag="region-concentration"]'));
      expect(badge).withContext('region-concentration badge must be visible').not.toBeNull();
    });

    it('when sector > 15 %, renders a sector-concentration flag badge', () => {
      // Technology = 75 % (3/4)
      const profiles = [
        makeProfile('Technology', 'United States'),
        makeProfile('Technology', 'United States'),
        makeProfile('Technology', 'United States'),
        makeProfile('Healthcare', 'Germany'),
      ];
      yfinanceSpy.getProfile.and.returnValues(...profiles.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['t1', 't2', 't3', 't4']);
      fixture.detectChanges();

      expect(
        fixture.debugElement.query(By.css('[data-flag="sector-concentration"]')),
      ).not.toBeNull();
    });

    it('when HHI >= 0.12 (N ≤ 8 equal-weight assets), renders an HHI flag badge', () => {
      // 2 assets → 1/2 = 0.5 >= 0.12
      const profiles = [
        makeProfile('Technology', 'United States'),
        makeProfile('Technology', 'United States'),
      ];
      yfinanceSpy.getProfile.and.returnValues(...profiles.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['t1', 't2']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="hhi"]'))).not.toBeNull();
    });

    it('when top-4 holdings >= 30 %, renders a top4-concentration flag badge', () => {
      // 2 assets → top-4 = 100 % > 30 %
      const profiles = [
        makeProfile('Technology', 'United States'),
        makeProfile('Technology', 'United States'),
      ];
      yfinanceSpy.getProfile.and.returnValues(...profiles.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['t1', 't2']);
      fixture.detectChanges();

      expect(
        fixture.debugElement.query(By.css('[data-flag="top4-concentration"]')),
      ).not.toBeNull();
    });

    it('when healthcare weight < 8 %, renders a healthcare flag badge', () => {
      // No healthcare assets
      yfinanceSpy.getProfile.and.returnValue(of(makeProfile('Technology', 'United States')));
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="healthcare"]'))).not.toBeNull();
    });

    it('when technology weight < 10 %, renders a technology flag badge', () => {
      // No technology assets
      yfinanceSpy.getProfile.and.returnValue(of(makeProfile('Healthcare', 'Germany')));
      fixture.componentRef.setInput('tickers', ['rog']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="technology"]'))).not.toBeNull();
    });

    it('when a major sector is absent, renders an absent-sector flag badge', () => {
      yfinanceSpy.getProfile.and.returnValue(of(makeProfile('Technology', 'United States')));
      fixture.componentRef.setInput('tickers', ['aapl']);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-flag="absent-sector"]'))).not.toBeNull();
    });

    // ── Regime escalation ────────────────────────────────────────────────

    it('when regime is CONTRACTION and a violation exists, flag badge has flag-escalated class', () => {
      macroSubject.next(makeCalibration('CONTRACTION'));
      yfinanceSpy.getProfile.and.returnValues(...ALL_US_PROFILES.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['aapl', 'msft', 'amzn']);
      fixture.detectChanges();

      const badge = fixture.debugElement.query(By.css('[data-flag="region-concentration"]'));
      expect(badge).not.toBeNull();
      expect(badge.nativeElement.classList)
        .withContext('CONTRACTION → badge must carry flag-escalated class')
        .toContain('flag-escalated');
    });

    it('when regime is not CONTRACTION, flag badges do not have flag-escalated class', () => {
      macroSubject.next(makeCalibration('EARLY_EXPANSION'));
      yfinanceSpy.getProfile.and.returnValues(...ALL_US_PROFILES.map(p => of(p)));
      fixture.componentRef.setInput('tickers', ['aapl', 'msft', 'amzn']);
      fixture.detectChanges();

      const badge = fixture.debugElement.query(By.css('[data-flag="region-concentration"]'));
      expect(badge).not.toBeNull();
      expect(badge.nativeElement.classList)
        .withContext('non-CONTRACTION → badge must NOT carry flag-escalated class')
        .not.toContain('flag-escalated');
    });

    it('when multiple violations coexist, each renders a separate flag badge', () => {
      // 2 US assets, all Technology → region + sector + HHI + top4 + healthcare + technology + absent-sector
      yfinanceSpy.getProfile.and.returnValues(
        of(makeProfile('Technology', 'United States')),
        of(makeProfile('Technology', 'United States')),
      );
      fixture.componentRef.setInput('tickers', ['t1', 't2']);
      fixture.detectChanges();

      const badges = fixture.debugElement.queryAll(By.css('[data-flag]'));
      expect(badges.length)
        .withContext('each violation must have its own badge')
        .toBeGreaterThan(1);
    });

  });

  // ======================================================================
  // Loading / error resilience
  // ======================================================================

  describe('resilience', () => {

    it('when getProfile returns null (fetch error), component does not crash', () => {
      yfinanceSpy.getProfile.and.returnValue(of(null));
      expect(() => {
        fixture.componentRef.setInput('tickers', ['bad-id']);
        fixture.detectChanges();
      }).not.toThrow();
    });

    it('when getMacroCalibration returns null (fetch error), component does not crash', () => {
      macroSpy.getMacroCalibration.and.returnValue(of(null));
      expect(() => {
        fixture.componentRef.setInput('tickers', ['aapl']);
        fixture.detectChanges();
      }).not.toThrow();
    });

    it('when tickers is empty, no charts are rendered', () => {
      fixture.componentRef.setInput('tickers', []);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('[data-chart="sector"]'))).toBeNull();
      expect(fixture.debugElement.query(By.css('[data-chart="region"]'))).toBeNull();
    });

  });
});
