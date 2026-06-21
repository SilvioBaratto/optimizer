import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { By } from '@angular/platform-browser';

import { AssetListComponent } from './asset-list';
import { Instrument } from '../../core/models/universe.model';

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

describe('AssetListComponent', () => {
  let fixture: ComponentFixture<AssetListComponent>;
  let component: AssetListComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AssetListComponent],
      providers: [provideZonelessChangeDetection()],
    }).compileComponents();

    fixture = TestBed.createComponent(AssetListComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('when assets input is empty, the empty placeholder is shown', () => {
    const host = fixture.nativeElement as HTMLElement;
    expect(host.textContent).toContain('No assets selected');
  });

  it('when assets input contains an instrument, its ticker is rendered', () => {
    fixture.componentRef.setInput('assets', [AAPL]);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('AAPL');
  });

  it('when assets input contains two instruments, both tickers are rendered', () => {
    fixture.componentRef.setInput('assets', [AAPL, MSFT]);
    fixture.detectChanges();
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('AAPL');
    expect(text).toContain('MSFT');
  });

  it('when the remove button is clicked, the remove output emits the instrument id', () => {
    fixture.componentRef.setInput('assets', [AAPL]);
    fixture.detectChanges();

    let emitted: string | undefined;
    component.remove.subscribe((id: string) => (emitted = id));

    const btn = fixture.debugElement.query(By.css('button'));
    btn.nativeElement.click();

    expect(emitted).toBe(AAPL.id);
  });

  it('when assets input transitions from populated to empty, the placeholder is shown', () => {
    fixture.componentRef.setInput('assets', [AAPL]);
    fixture.detectChanges();

    fixture.componentRef.setInput('assets', []);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('No assets selected');
  });
});
