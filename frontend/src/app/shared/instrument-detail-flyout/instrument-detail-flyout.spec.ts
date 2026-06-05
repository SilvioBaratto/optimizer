import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { InstrumentDetailFlyoutComponent } from './instrument-detail-flyout';
import { environment } from '../../../environments/environment';
import { ICON_PROVIDER } from '../../icons';

const BASE = `${environment.apiUrl}yfinance-data/instruments`;

describe('InstrumentDetailFlyoutComponent', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function createFixture(instrumentId: string | null) {
    const fx = TestBed.createComponent(InstrumentDetailFlyoutComponent);
    fx.componentRef.setInput('instrumentId', instrumentId);
    fx.detectChanges();
    return fx;
  }

  it('fetches profile + prices + recommendations when instrumentId is set', () => {
    const fx = createFixture('abc');

    http.expectOne(`${BASE}/abc/profile`).flush({
      id: 'abc',
      instrument_id: 'abc',
      short_name: 'Apple Inc',
      long_name: 'Apple Inc',
      current_price: 190.5,
      market_cap: 3_000_000_000_000,
      sector: 'Technology',
      industry: 'Consumer Electronics',
      currency: 'USD',
      trailing_pe: 28.5,
    });
    http
      .expectOne((r) => r.url === `${BASE}/abc/prices`)
      .flush([
        {
          id: 'p1', instrument_id: 'abc', date: '2024-01-01',
          open: 184, high: 186, low: 183, close: 185, volume: 1, dividends: 0,
          stock_splits: 0, created_at: '', updated_at: '',
        },
        {
          id: 'p2', instrument_id: 'abc', date: '2024-01-02',
          open: 185, high: 189, low: 185, close: 188, volume: 1, dividends: 0,
          stock_splits: 0, created_at: '', updated_at: '',
        },
      ]);
    http.expectOne(`${BASE}/abc/recommendations`).flush([
      {
        id: 'r1', instrument_id: 'abc', period: '0m',
        strong_buy: 5, buy: 20, hold: 10, sell: 2, strong_sell: 1,
        created_at: '', updated_at: '',
      },
    ]);

    expect(fx.componentInstance.profile()).toBeTruthy();
    expect(fx.componentInstance.profile()!.long_name).toBe('Apple Inc');
    expect(fx.componentInstance.prices().length).toBe(2);
    expect(fx.componentInstance.recommendations().length).toBe(1);
  });

  it('does not make requests when instrumentId is null', () => {
    const fx = createFixture(null);
    http.expectNone((r) => r.url.startsWith(BASE));
    expect(fx.componentInstance.profile()).toBeNull();
  });

  it('re-fetches when instrumentId input changes', () => {
    const fx = createFixture('abc');

    http.expectOne(`${BASE}/abc/profile`).flush(null);
    http.expectOne((r) => r.url === `${BASE}/abc/prices`).flush([]);
    http.expectOne(`${BASE}/abc/recommendations`).flush([]);

    fx.componentRef.setInput('instrumentId', 'xyz');
    fx.detectChanges();

    const profileReq = http.expectOne(`${BASE}/xyz/profile`);
    expect(profileReq.request.method).toBe('GET');
    profileReq.flush(null);
    http.expectOne((r) => r.url === `${BASE}/xyz/prices`).flush([]);
    http.expectOne(`${BASE}/xyz/recommendations`).flush([]);
    expect(fx.componentInstance.instrumentId()).toBe('xyz');
  });

  it('exposes a close() output that fires on user close', () => {
    const fx = createFixture('abc');
    let closed = false;
    fx.componentInstance.closed.subscribe(() => (closed = true));

    http.expectOne(`${BASE}/abc/profile`).flush(null);
    http.expectOne((r) => r.url === `${BASE}/abc/prices`).flush([]);
    http.expectOne(`${BASE}/abc/recommendations`).flush([]);

    fx.componentInstance.onClose();
    expect(closed).toBe(true);
  });

  it('emits closed when the user presses Escape while the flyout is open', () => {
    const fx = createFixture('abc');
    let closed = false;
    fx.componentInstance.closed.subscribe(() => (closed = true));

    http.expectOne(`${BASE}/abc/profile`).flush(null);
    http.expectOne((r) => r.url === `${BASE}/abc/prices`).flush([]);
    http.expectOne(`${BASE}/abc/recommendations`).flush([]);

    const event = new KeyboardEvent('keydown', { key: 'Escape' });
    document.dispatchEvent(event);

    expect(closed).toBe(true);
  });

  it('ignores Escape key when no instrument is selected', () => {
    const fx = createFixture(null);
    let closed = false;
    fx.componentInstance.closed.subscribe(() => (closed = true));

    const event = new KeyboardEvent('keydown', { key: 'Escape' });
    document.dispatchEvent(event);

    expect(closed).toBe(false);
  });

  it('when the instrument switches mid-flight, the newer response wins and the stale one is cancelled', () => {
    const fx = createFixture('old');

    // 'old' requests are in flight (not yet flushed).
    const oldProfile = http.expectOne(`${BASE}/old/profile`);
    http.expectOne((r) => r.url === `${BASE}/old/prices`);
    http.expectOne(`${BASE}/old/recommendations`);

    // Switch instrument before 'old' resolves — switchMap must cancel it.
    fx.componentRef.setInput('instrumentId', 'new');
    fx.detectChanges();
    expect(oldProfile.cancelled).toBe(true);

    http
      .expectOne(`${BASE}/new/profile`)
      .flush({ id: 'new', instrument_id: 'new', long_name: 'New Co' });
    http.expectOne((r) => r.url === `${BASE}/new/prices`).flush([]);
    http.expectOne(`${BASE}/new/recommendations`).flush([]);

    expect(fx.componentInstance.profile()!.long_name).toBe('New Co');
  });

  it('falls back to empty arrays when endpoints return errors', () => {
    const fx = createFixture('abc');

    http
      .expectOne(`${BASE}/abc/profile`)
      .error(new ProgressEvent('network'), { status: 500 });
    http
      .expectOne((r) => r.url === `${BASE}/abc/prices`)
      .error(new ProgressEvent('network'), { status: 500 });
    http
      .expectOne(`${BASE}/abc/recommendations`)
      .error(new ProgressEvent('network'), { status: 500 });

    expect(fx.componentInstance.profile()).toBeNull();
    expect(fx.componentInstance.prices()).toEqual([]);
    expect(fx.componentInstance.recommendations()).toEqual([]);
  });
});
