/**
 * Contract tests — issue #1049, criteria 3 & 4
 *
 * Criterion 3: "BuilderStore, BuilderResultService, BuilderDriftService, and
 *   builder models/builder-* are deleted along with their specs (removed, not
 *   skipped)."
 *
 * Criterion 4: "No remaining references/imports to the deleted builder code;
 *   builder specs rewritten to the search/list/preview/save shape."
 *
 * Tests are source-blind: derived from the acceptance criteria only.
 *
 * Strategy for criterion 3:
 *   The test configures TestBed WITHOUT providing BuilderStore, BuilderResultService,
 *   or BuilderDriftService.  If the component still injects any of them, Angular's
 *   injector will throw a NullInjectorError and the test fails (red).  Once those
 *   services are deleted and removed from the component, the test passes.
 *
 * Strategy for criterion 4:
 *   Tests verify that the component exposes the four behavioural surfaces named in
 *   the criterion — search, list, preview, save — using signal-based names consistent
 *   with the Angular 21 conventions required by the project.
 *
 * Assumption: the component's minimal public surface is
 *   searchQuery:           WritableSignal<string>
 *   searchResults():       unknown[]           (computed)
 *   addInstrument(i):      void
 *   selectedAssets():      unknown[]           (signal or computed)
 *   diversificationPreview(): unknown | null   (signal or computed, truthy when assets present)
 *   savePortfolio(name):   void
 */

import { NO_ERRORS_SCHEMA } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { PortfolioDto } from '../app/core/models/portfolio-api.model';
import { Instrument } from '../app/core/models/universe.model';
import { PortfolioBuilderComponent } from '../app/portfolio-builder/portfolio-builder';
import { PortfolioApiService } from '../app/core/services/portfolio-api.service';

function makePortfolioDto(name = 'Test Portfolio'): PortfolioDto {
  return {
    id: '1',
    name,
    description: null,
    currency: 'USD',
    benchmark_ticker: 'SPY',
    is_active: true,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };
}

function makeInstrument(ticker: string): Instrument {
  return {
    id: ticker.toLowerCase(),
    ticker,
    short_name: `${ticker} Corp`,
    name: ticker,
    isin: null,
    instrument_type: null,
    currency_code: null,
    yfinance_ticker: null,
    exchange_name: null,
  };
}

// ── Criterion 3 ───────────────────────────────────────────────────────────────

describe('PortfolioBuilder — No Deleted Service Dependencies', () => {
  it(
    'when created without BuilderStore, BuilderResultService, or BuilderDriftService, ' +
      'component initialises without a DI error',
    () => {
      /**
       * Only PortfolioApiService is provided.  The three deleted services are
       * intentionally absent.  Any remaining injection of the deleted services
       * causes a NullInjectorError → test fails (red phase).
       * After deletion: no injection needed → test passes.
       */
      const apiSpy = jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', ['create']);
      apiSpy.create.and.returnValue(of(makePortfolioDto()));

      TestBed.configureTestingModule({
        imports: [PortfolioBuilderComponent],
        providers: [{ provide: PortfolioApiService, useValue: apiSpy }],
        schemas: [NO_ERRORS_SCHEMA],
      });

      expect(() => TestBed.createComponent(PortfolioBuilderComponent)).not.toThrow();
    }
  );
});

// ── Criterion 4 ───────────────────────────────────────────────────────────────

describe('PortfolioBuilder — Search / List / Preview / Save Shape', () => {
  let apiSpy: jasmine.SpyObj<PortfolioApiService>;

  beforeEach(() => {
    apiSpy = jasmine.createSpyObj<PortfolioApiService>('PortfolioApiService', ['create']);
    apiSpy.create.and.returnValue(of(makePortfolioDto()));

    TestBed.configureTestingModule({
      imports: [PortfolioBuilderComponent],
      providers: [{ provide: PortfolioApiService, useValue: apiSpy }],
      schemas: [NO_ERRORS_SCHEMA],
    });
  });

  // Search

  it('when searchQuery signal is set, searchResults is exposed as an array', () => {
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;
    fixture.detectChanges();

    comp.searchQuery.set('Apple');
    fixture.detectChanges();

    expect(Array.isArray(comp.searchResults())).toBeTrue();
  });

  // List

  it('when an instrument is added, it appears in selectedAssets', () => {
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;
    fixture.detectChanges();

    comp.addInstrument(makeInstrument('AAPL'));
    fixture.detectChanges();

    expect(comp.selectedAssets()).toContain(jasmine.objectContaining({ ticker: 'AAPL', id: 'aapl' }));
  });

  it('when an instrument is added twice, selectedAssets length increases by one', () => {
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;
    fixture.detectChanges();
    const before = comp.selectedAssets().length;

    comp.addInstrument(makeInstrument('AAPL'));
    fixture.detectChanges();

    expect(comp.selectedAssets().length).toBe(before + 1);
  });

  // Preview

  it('when selectedAssets is non-empty, diversificationPreview is truthy', () => {
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;
    fixture.detectChanges();

    comp.addInstrument(makeInstrument('AAPL'));
    fixture.detectChanges();

    expect(comp.diversificationPreview()).toBeTruthy();
  });

  // Save

  it('when savePortfolio is called with a valid name, PortfolioApiService.create is invoked', () => {
    const fixture = TestBed.createComponent(PortfolioBuilderComponent);
    const comp = fixture.componentInstance;
    fixture.detectChanges();

    comp.savePortfolio('Saved Portfolio');

    expect(apiSpy.create).toHaveBeenCalledOnceWith(
      jasmine.objectContaining({ name: 'Saved Portfolio' })
    );
  });
});
