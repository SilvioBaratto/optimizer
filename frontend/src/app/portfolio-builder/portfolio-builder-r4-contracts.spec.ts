/**
 * R4 contracts — Portfolio Builder shell (issue #1047)
 *
 * Tests the PortfolioBuilderComponent shell in isolation.
 * C1 (type-ahead search) is covered by asset-search/asset-search.spec.ts.
 * This file covers:
 *   C2 – add to selected-asset list; remove from list; no duplicates
 *   C3 – no BuilderStore / BuilderResultService / BuilderDriftService providers;
 *         pipeline-stepper coupling absent; signal surface exposed
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection } from '@angular/core';
import { of } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { UniverseService } from '../core/services/universe.service';
import { YfinanceService } from '../core/services/yfinance.service';
import { MacroIntelligenceService } from '../macro-intelligence/macro-intelligence.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { Instrument } from '../core/models/universe.model';

function makeInstrument(ticker: string): Instrument {
  return {
    id: ticker.toLowerCase(),
    ticker,
    short_name: `${ticker} Corp`,
    name: null,
    isin: null,
    instrument_type: null,
    currency_code: null,
    yfinance_ticker: null,
    exchange_name: null,
  };
}

const AAPL = makeInstrument('AAPL');
const MSFT = makeInstrument('MSFT');
const GOOGL = makeInstrument('GOOGL');

describe('PortfolioBuilderComponent – R4 shell contracts', () => {
  let fixture: ComponentFixture<PortfolioBuilderComponent>;
  let component: PortfolioBuilderComponent;

  beforeEach(async () => {
    const universeSpy = jasmine.createSpyObj<UniverseService>('UniverseService', [
      'searchTickers',
    ]);
    universeSpy.searchTickers.and.returnValue(of({ items: [], total: 0, page: 1, page_size: 15 }));

    const yfinanceSpy = jasmine.createSpyObj<YfinanceService>('YfinanceService', ['getProfile']);
    yfinanceSpy.getProfile.and.returnValue(of(null));

    const macroSpy = jasmine.createSpyObj<MacroIntelligenceService>(
      'MacroIntelligenceService', ['getMacroCalibration'],
    );
    macroSpy.getMacroCalibration.and.returnValue(of(null));

    const apiSpy = jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', ['create']);

    await TestBed.configureTestingModule({
      imports: [PortfolioBuilderComponent],
      // NO_ERRORS_SCHEMA suppresses unknown-element errors from child components
      // so this suite focuses on the shell's own behaviour.
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        { provide: UniverseService,          useValue: universeSpy },
        { provide: YfinanceService,          useValue: yfinanceSpy },
        { provide: MacroIntelligenceService, useValue: macroSpy    },
        { provide: PortfolioApiService,      useValue: apiSpy      },
        // BuilderStore, BuilderResultService, BuilderDriftService intentionally absent.
        // A NullInjectorError in beforeEach would be the RED signal for C3.
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(PortfolioBuilderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  // =========================================================================
  // C2 — add / remove / no duplicates
  // =========================================================================

  it('when addAsset is called with an instrument, it appears in selectedAssets', () => {
    component.addAsset(AAPL);
    expect(component.selectedAssets()).toContain(AAPL);
  });

  it('when removeAsset is called with the instrument id, it is absent from selectedAssets', () => {
    component.addAsset(AAPL);
    component.removeAsset(AAPL.id);
    expect(component.selectedAssets()).not.toContain(AAPL);
  });

  it('when two distinct instruments are added, both appear in selectedAssets', () => {
    component.addAsset(AAPL);
    component.addAsset(MSFT);
    expect(component.selectedAssets()).toContain(AAPL);
    expect(component.selectedAssets()).toContain(MSFT);
  });

  it('when the first of two instruments is removed, the second remains in selectedAssets', () => {
    component.addAsset(AAPL);
    component.addAsset(MSFT);
    component.removeAsset(AAPL.id);
    expect(component.selectedAssets()).not.toContain(AAPL);
    expect(component.selectedAssets()).toContain(MSFT);
  });

  it('when removeAsset is called with an id not in the list, no error is thrown and the list is unchanged', () => {
    component.addAsset(AAPL);
    expect(() => component.removeAsset('nonexistent-id')).not.toThrow();
    expect(component.selectedAssets().length).toBe(1);
  });

  it('when the same instrument is added twice, selectedAssets contains it exactly once', () => {
    component.addAsset(AAPL);
    component.addAsset(AAPL);
    const count = component.selectedAssets().filter((a) => a.id === AAPL.id).length;
    expect(count).toBe(1);
  });

  // ── Property: no-duplicates invariant (parameterised over N additions) ────
  [2, 3, 5, 10].forEach((n) => {
    it(`when the same instrument is added ${n} times, selectedAssets contains it exactly once`, () => {
      for (let i = 0; i < n; i++) component.addAsset(GOOGL);
      const count = component.selectedAssets().filter((a) => a.id === GOOGL.id).length;
      expect(count).toBe(1);
    });
  });

  // ── Property: id-uniqueness invariant ─────────────────────────────────────
  it('when instruments are added with repeats, no two selectedAssets entries share an id', () => {
    [AAPL, MSFT, GOOGL, AAPL, MSFT].forEach((i) => component.addAsset(i));
    const ids = component.selectedAssets().map((a) => a.id);
    expect(ids.length).toBe(new Set(ids).size);
  });

  // =========================================================================
  // C3 — no forbidden providers; signals; no pipeline-stepper
  // =========================================================================

  it('when created without BuilderStore provided, the component initialises without an injection error', () => {
    // Reaching this line proves no forbidden provider is required by the shell.
    expect(component).toBeTruthy();
  });

  it('when created, selectedAssets is a callable signal returning an array', () => {
    expect(typeof component.selectedAssets).toBe('function');
    expect(Array.isArray(component.selectedAssets())).toBeTrue();
  });

  it('when the template is rendered, no app-pipeline-stepper element is present', () => {
    const el = fixture.nativeElement.querySelector('app-pipeline-stepper');
    expect(el).toBeNull();
  });

  it('when the template is rendered, the search region is present', () => {
    const el = fixture.nativeElement.querySelector('[data-region="search"]');
    expect(el).not.toBeNull();
  });

  it('when the template is rendered, the list region is present', () => {
    const el = fixture.nativeElement.querySelector('[data-region="list"]');
    expect(el).not.toBeNull();
  });
});
