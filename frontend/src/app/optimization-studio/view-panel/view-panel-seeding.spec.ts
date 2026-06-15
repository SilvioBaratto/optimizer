/**
 * Ticker-seeding spec for ViewPanelComponent — issue #1023.
 *
 * Criteria under test:
 *   "view-panel.ts adds tickers = input<string[]>() and uses it for
 *    /views/generate, /views/opinion-pool, /views/entropy-pooling request bodies,
 *    falling back to its current default when the input is empty."
 *
 *   "A portfolio change that re-seeds the parent universe is reflected in the
 *    next /views/* POST body (request tickers equals Object.keys(snapshot.weights))."
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ViewPanelComponent } from './view-panel';
import { environment } from '../../../environments/environment';

const API = environment.apiUrl;
const GENERATE_URL = `${API}views/generate`;
const OPINION_URL = `${API}views/opinion-pool`;
const ENTROPY_URL = `${API}views/entropy-pooling`;

/** Snapshot tickers simulating Object.keys(snapshot.weights) from portfolio. */
const SNAPSHOT_TICKERS = ['TSLA', 'NVDA', 'META', 'AMZN'];

const BL_STUB = {
  nViews: 0,
  nAssets: 0,
  viewStrings: [],
  p: [],
  q: [],
  viewConfidences: [],
  idzorekAlphas: {},
  views: [],
  rationale: '',
  tickersWithData: [],
  tickersMissingData: [],
};

const OPINION_STUB = {
  nExperts: 0,
  tickers: [],
  tickersWithData: [],
  tickersMissingData: [],
  experts: [],
  icWeights: [],
  poolingType: 'linear',
  totalViews: 0,
};

describe('ViewPanelComponent — ticker seeding from parent [tickers] input', () => {
  let fixture: ComponentFixture<ViewPanelComponent>;
  let comp: ViewPanelComponent;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({
      imports: [ViewPanelComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(ViewPanelComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  // ── POST /views/generate (Black-Litterman) ─────────────────────────────────

  describe('POST /views/generate', () => {
    it('when tickers input equals snapshot weights keys, the POST body carries exactly those tickers', () => {
      fixture.componentRef.setInput('tickers', SNAPSHOT_TICKERS);
      fixture.detectChanges();

      comp.runBlackLitterman();

      const req = http.expectOne(GENERATE_URL);
      expect(req.request.body['tickers']).toEqual(SNAPSHOT_TICKERS);
      req.flush(BL_STUB);
    });

    it('when tickers input is empty, the POST body falls back to the non-empty default from tickersRaw', () => {
      fixture.componentRef.setInput('tickers', []);
      fixture.detectChanges();

      comp.runBlackLitterman();

      const req = http.expectOne(GENERATE_URL);
      const tickers: string[] = req.request.body['tickers'];
      expect(tickers.length)
        .withContext('Expected fallback tickers to be non-empty')
        .toBeGreaterThan(0);
      req.flush(BL_STUB);
    });
  });

  // ── POST /views/opinion-pool ────────────────────────────────────────────────

  describe('POST /views/opinion-pool', () => {
    it('when tickers input equals snapshot weights keys, the POST body carries exactly those tickers', () => {
      fixture.componentRef.setInput('tickers', SNAPSHOT_TICKERS);
      fixture.detectChanges();

      // Default personasRaw has 3 entries so the personas-length guard clears.
      comp.runOpinionPool();

      const req = http.expectOne(OPINION_URL);
      expect(req.request.body['tickers']).toEqual(SNAPSHOT_TICKERS);
      req.flush(OPINION_STUB);
    });

    it('when tickers input is empty, the POST body falls back to the non-empty default from tickersRaw', () => {
      fixture.componentRef.setInput('tickers', []);
      fixture.detectChanges();

      comp.runOpinionPool();

      const req = http.expectOne(OPINION_URL);
      const tickers: string[] = req.request.body['tickers'];
      expect(tickers.length)
        .withContext('Expected fallback tickers to be non-empty')
        .toBeGreaterThan(0);
      req.flush(OPINION_STUB);
    });
  });

  // ── POST /views/entropy-pooling ─────────────────────────────────────────────

  describe('POST /views/entropy-pooling', () => {
    it('when tickers input equals snapshot weights keys, the POST body carries exactly those tickers', () => {
      fixture.componentRef.setInput('tickers', SNAPSHOT_TICKERS);
      fixture.detectChanges();

      // runEntropyPooling has no views-content guard in its method body;
      // the disabled state is enforced at template level only.
      comp.runEntropyPooling();

      const req = http.expectOne(ENTROPY_URL);
      expect(req.request.body['tickers']).toEqual(SNAPSHOT_TICKERS);
      req.flush({ tickers: SNAPSHOT_TICKERS, mu: SNAPSHOT_TICKERS.map(() => 0.05) });
    });

    it('when tickers input is empty, the POST body falls back to the non-empty default from tickersRaw', () => {
      fixture.componentRef.setInput('tickers', []);
      fixture.detectChanges();

      comp.runEntropyPooling();

      const req = http.expectOne(ENTROPY_URL);
      const tickers: string[] = req.request.body['tickers'];
      expect(tickers.length)
        .withContext('Expected fallback tickers to be non-empty')
        .toBeGreaterThan(0);
      req.flush({ tickers: ['AAPL', 'MSFT'], mu: [0.05, 0.08] });
    });
  });

  // ── Portfolio re-seed propagation ───────────────────────────────────────────

  describe('portfolio re-seed propagation', () => {
    it('when tickers input changes (portfolio re-seeds), the next /views/generate POST carries the new tickers', () => {
      // Criterion: "A portfolio change that re-seeds the parent universe is reflected
      // in the next /views/* POST body (request tickers equals Object.keys(snapshot.weights))."
      const INITIAL = ['AAPL', 'MSFT', 'GOOGL'];
      const UPDATED = ['NVDA', 'AMD', 'INTC', 'QCOM'];

      fixture.componentRef.setInput('tickers', INITIAL);
      fixture.detectChanges();

      comp.runBlackLitterman();
      http.expectOne(GENERATE_URL).flush(BL_STUB);

      // Parent re-seeds from new portfolio snapshot
      fixture.componentRef.setInput('tickers', UPDATED);
      fixture.detectChanges();

      comp.runBlackLitterman();
      const req2 = http.expectOne(GENERATE_URL);
      expect(req2.request.body['tickers'])
        .withContext('Expected the updated portfolio tickers after re-seed')
        .toEqual(UPDATED);
      req2.flush(BL_STUB);
    });

    it('when tickers input changes, the next /views/opinion-pool POST carries the new tickers', () => {
      const INITIAL = ['JPM', 'GS', 'MS'];
      const UPDATED = ['V', 'MA', 'AXP', 'BRK'];

      fixture.componentRef.setInput('tickers', INITIAL);
      fixture.detectChanges();
      comp.runOpinionPool();
      http.expectOne(OPINION_URL).flush(OPINION_STUB);

      fixture.componentRef.setInput('tickers', UPDATED);
      fixture.detectChanges();
      comp.runOpinionPool();

      const req2 = http.expectOne(OPINION_URL);
      expect(req2.request.body['tickers']).toEqual(UPDATED);
      req2.flush(OPINION_STUB);
    });
  });
});
