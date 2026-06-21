/**
 * Asset-search component — C1 contracts (issue #1047)
 *
 * Tests:
 *   - type-ahead search routes queries through UniverseService.searchTickers
 *   - results signal reflects what the service emits
 *   - clearing the query resets results to []
 *   - clicking a result emits the add output
 *
 * Note: debounce lives inside UniverseService.searchTickers. The spy here
 * bypasses debounce intentionally (unit test scope). Timing is covered by
 * the service's own spec.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { By } from '@angular/platform-browser';
import { Observable, of, switchMap } from 'rxjs';

import { AssetSearchComponent } from './asset-search';
import { UniverseService } from '../../core/services/universe.service';
import { Instrument, InstrumentList } from '../../core/models/universe.model';

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

function makeList(items: Instrument[]): InstrumentList {
  return { items, total: items.length, page: 1, page_size: 15 };
}

const AAPL = makeInstrument('AAPL');
const MSFT = makeInstrument('MSFT');

/** Spy whose searchTickers passes the query Observable through synchronously. */
function makeSpy(queryToItems: (q: string) => Instrument[]): jasmine.SpyObj<UniverseService> {
  const spy = jasmine.createSpyObj<UniverseService>('UniverseService', ['searchTickers']);
  spy.searchTickers.and.callFake((q$: Observable<string>) =>
    q$.pipe(switchMap((q) => of(makeList(queryToItems(q))))),
  );
  return spy;
}

describe('AssetSearchComponent – C1 type-ahead search contracts', () => {
  let fixture: ComponentFixture<AssetSearchComponent>;
  let component: AssetSearchComponent;
  let universeSvc: jasmine.SpyObj<UniverseService>;

  beforeEach(async () => {
    universeSvc = makeSpy((q) => (q === 'AAPL' ? [AAPL] : q === 'MSFT' ? [MSFT] : []));

    await TestBed.configureTestingModule({
      imports: [AssetSearchComponent],
      providers: [
        provideZonelessChangeDetection(),
        { provide: UniverseService, useValue: universeSvc },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AssetSearchComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  // ── Search wiring ──────────────────────────────────────────────────────────

  it('when the component is created, searchTickers is called with an Observable', () => {
    expect(universeSvc.searchTickers).toHaveBeenCalled();
    const arg = universeSvc.searchTickers.calls.first().args[0];
    expect(arg).toBeInstanceOf(Observable);
  });

  it('when the query control is set to "AAPL", results() contains the AAPL instrument', () => {
    component.queryCtrl.setValue('AAPL');
    fixture.detectChanges();
    expect(component.results()).toContain(AAPL);
  });

  it('when the query is set to "MSFT", results() contains the MSFT instrument', () => {
    component.queryCtrl.setValue('MSFT');
    fixture.detectChanges();
    expect(component.results()).toContain(MSFT);
  });

  it('when the query is cleared after a search, results() resets to []', () => {
    component.queryCtrl.setValue('AAPL');
    fixture.detectChanges();

    component.queryCtrl.setValue('');
    fixture.detectChanges();

    expect(component.results()).toEqual([]);
  });

  it('when a second distinct query replaces the first, results() reflects only the new response', () => {
    component.queryCtrl.setValue('AAPL');
    fixture.detectChanges();

    component.queryCtrl.setValue('MSFT');
    fixture.detectChanges();

    expect(component.results()).toContain(MSFT);
    expect(component.results()).not.toContain(AAPL);
  });

  it('when the query is unknown, results() is empty', () => {
    component.queryCtrl.setValue('XYZ_UNKNOWN');
    fixture.detectChanges();
    expect(component.results()).toEqual([]);
  });

  // ── Add output ─────────────────────────────────────────────────────────────

  it('when a result button is clicked, the add output emits that instrument', () => {
    component.queryCtrl.setValue('AAPL');
    fixture.detectChanges();

    let emitted: Instrument | undefined;
    component.add.subscribe((i: Instrument) => (emitted = i));

    const btn = fixture.debugElement.query(By.css('button'));
    btn.nativeElement.click();

    expect(emitted).toBe(AAPL);
  });

  // ── Signal contract ────────────────────────────────────────────────────────

  it('when created, results is a callable signal returning an array', () => {
    expect(typeof component.results).toBe('function');
    expect(Array.isArray(component.results())).toBeTrue();
  });

  it('when created with no user input, results() starts as []', () => {
    expect(component.results()).toEqual([]);
  });
});
