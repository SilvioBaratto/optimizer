import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { UniversePanelComponent } from './universe-panel';
import { SEARCH_DEBOUNCE_MS } from '../../services/universe.service';
import { environment } from '../../../environments/environment';
import { ICON_PROVIDER } from '../../icons';

const API = environment.apiUrl;

describe('UniversePanelComponent', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: SEARCH_DEBOUNCE_MS, useValue: 0 },
        ICON_PROVIDER,
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function createComponent() {
    const fx = TestBed.createComponent(UniversePanelComponent);
    fx.detectChanges();
    return fx;
  }

  it('POSTs /universe/screen with the selected preset when runScreen fires', () => {
    const fx = createComponent();
    fx.componentInstance.setPreset('developed_markets');
    fx.componentInstance.runScreen();

    const req = http.expectOne(`${API}universe/screen`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ preset: 'developed_markets' });
    req.flush({
      passingTickers: ['AAPL'],
      totalScreened: 1,
      diagnostics: {
        AAPL: { market_cap: true, addv_3m: true, trading_frequency: true, price_us: true },
      },
    });

    expect(fx.componentInstance.passingCount()).toBe(1);
    expect(fx.componentInstance.screenRows()).toEqual([
      {
        ticker: 'AAPL',
        marketCap: '✓',
        addv3m: '✓',
        tradingFrequency: '✓',
        ipoSeasoning: '—',
        priceFloor: '✓',
      },
    ]);
  });

  it('debounces ticker search and queries /universe/instruments', (done) => {
    const fx = createComponent();
    fx.componentInstance.onSearchInput('A');
    fx.componentInstance.onSearchInput('AP');
    fx.componentInstance.onSearchInput('APPL');

    setTimeout(() => {
      const reqs = http.match((r) => r.url === `${API}universe/instruments`);
      expect(reqs.length).toBe(1);
      expect(reqs[0].request.params.get('search')).toBe('APPL');
      reqs[0].flush({
        items: [
          {
            id: 'i1', ticker: 'AAPL', name: 'Apple', exchange: 'NASDAQ',
            type: 'equity', currency: 'USD', isin: 'US0378331005',
          },
        ],
        total: 1, page: 1, page_size: 15,
      });

      expect(fx.componentInstance.searchResults().length).toBe(1);
      done();
    }, 10);
  });

  it('surfaces the error when /universe/screen returns 422', () => {
    const fx = createComponent();
    fx.componentInstance.runScreen();

    http
      .expectOne(`${API}universe/screen`)
      .flush({ detail: 'bad config' }, { status: 422, statusText: 'Unprocessable' });

    expect(fx.componentInstance.screenError()).toBeTruthy();
    expect(fx.componentInstance.screenLoading()).toBe(false);
  });

  it('emits tickerSelected when a search result is clicked', () => {
    const fx = createComponent();
    let emitted: string | undefined;
    fx.componentInstance.tickerSelected.subscribe((inst) => (emitted = inst.id));

    fx.componentInstance.onTickerClick({
      id: 'abc', ticker: 'AAPL', name: 'Apple', exchange: 'NASDAQ',
      type: 'equity', currency: 'USD', isin: 'US0378331005',
    });

    expect(emitted).toBe('abc');
  });

  it('exposes the 4 backend screen presets', () => {
    const fx = createComponent();
    expect(fx.componentInstance.presets).toEqual([
      'developed_markets',
      'broad_universe',
      'small_cap',
      'large_cap',
    ]);
  });
});
