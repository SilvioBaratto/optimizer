/**
 * factor-research-score-select-chain.spec.ts — issue #1025
 *
 * Tests: /factors/score and /factors/select round-trips populate result signals
 * and render; loading flags toggle; error paths set the right panelErrors key
 * with a visible banner.
 *
 * Criteria covered:
 *   - scoreLoading toggles true on call, false on success/error
 *   - selectLoading toggles true on call, false on success/error
 *   - scoreResult() populated from POST /factors/score success response
 *   - selectResult() populated from POST /factors/select success response
 *   - panelErrors['score'] set on 500; not set on success
 *   - panelErrors['select'] set on 500; not set on success
 *   - score/select errors do not cross-contaminate each other's panelErrors key
 *   - score/select tab shows [role="alert"] banner on error
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';
import { environment } from '../../environments/environment';
import type { FactorScoreRequest, FactorSelectRequest } from './factor.model';

const API = environment.apiUrl;
const SCORE_URL = `${API}factors/score`;
const SELECT_URL = `${API}factors/select`;

const SCORE_PAYLOAD: FactorScoreRequest = {
  tickers: ['AAPL', 'MSFT', 'GOOGL'],
  score_date: '2024-12-31',
  composite_method: 'equal_weight',
};

const SELECT_PAYLOAD: FactorSelectRequest = {
  tickers: ['AAPL', 'MSFT', 'GOOGL'],
  start_date: '2023-01-01',
  end_date: '2024-12-31',
};

const SCORE_RESPONSE = {
  score_date: '2024-12-31',
  scores: { AAPL: 0.85, MSFT: 0.72, GOOGL: 0.61 },
  group_contributions: { momentum: 0.3, value: 0.5 },
};

const SELECT_RESPONSE = {
  selected_tickers: ['AAPL', 'MSFT'],
  count: 2,
  turnover: 0.1,
  buffer_zone: { entered: ['AAPL'], exited: [] },
};

describe('FactorResearchComponent — score + select chain (issue #1025)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let http: HttpTestingController;
  let el: HTMLElement;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [FactorResearchComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(FactorResearchComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── score: loading flag ──────────────────────────────────────────────────────

  it('when onRunScore() is called, scoreLoading() becomes true before response arrives', () => {
    comp.onRunScore(SCORE_PAYLOAD);

    expect(comp.scoreLoading())
      .withContext('scoreLoading must be true while request is in-flight')
      .toBeTrue();

    http.expectOne(SCORE_URL).flush(SCORE_RESPONSE);
  });

  it('when POST /factors/score succeeds, scoreLoading() becomes false', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush(SCORE_RESPONSE);

    expect(comp.scoreLoading())
      .withContext('scoreLoading must be false after successful response')
      .toBeFalse();
  });

  it('when onRunScore() is called, panelErrors["score"] is cleared to null', () => {
    comp.panelErrors.update((p) => ({ ...p, score: 'prior error' }));
    comp.onRunScore(SCORE_PAYLOAD);

    expect(comp.panelErrors()['score'])
      .withContext('panelErrors["score"] must be cleared on re-run')
      .toBeNull();

    http.expectOne(SCORE_URL).flush(SCORE_RESPONSE);
  });

  // ── score: success path ──────────────────────────────────────────────────────

  it('when POST /factors/score succeeds, scoreResult() is populated', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush(SCORE_RESPONSE);

    expect(comp.scoreResult()).not.toBeNull();
    expect(comp.scoreResult()?.scores['AAPL']).toBe(0.85);
    expect(comp.scoreResult()?.scores['MSFT']).toBe(0.72);
  });

  it('when POST /factors/score succeeds, panelErrors["score"] stays null', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush(SCORE_RESPONSE);

    expect(comp.panelErrors()['score'] ?? null)
      .withContext('no error on success')
      .toBeNull();
  });

  it('when scoreResult() is set, score tab renders app-score-panel', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush(SCORE_RESPONSE);

    comp.activeTab.set('score');
    fixture.detectChanges();

    expect(el.querySelector('app-score-panel'))
      .withContext('app-score-panel must be present on score tab')
      .not.toBeNull();
  });

  it('when POST /factors/score is called, request body contains the tickers array', () => {
    comp.onRunScore(SCORE_PAYLOAD);

    const req = http.expectOne(SCORE_URL);
    expect((req.request.body as FactorScoreRequest).tickers).toEqual(SCORE_PAYLOAD.tickers);
    req.flush(SCORE_RESPONSE);
  });

  it('when POST /factors/score is called, HTTP method is POST', () => {
    comp.onRunScore(SCORE_PAYLOAD);

    const req = http.expectOne(SCORE_URL);
    expect(req.request.method).toBe('POST');
    req.flush(SCORE_RESPONSE);
  });

  // ── score: error path ────────────────────────────────────────────────────────

  it('when POST /factors/score returns 500, panelErrors["score"] is set', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.panelErrors()['score'])
      .withContext('panelErrors["score"] must be truthy on 500')
      .toBeTruthy();
  });

  it('when POST /factors/score errors, scoreLoading() becomes false', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.scoreLoading())
      .withContext('scoreLoading must be false after error')
      .toBeFalse();
  });

  it('when POST /factors/score errors, scoreResult() stays null', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.scoreResult()).toBeNull();
  });

  it('when POST /factors/score errors, score tab shows [role="alert"] banner', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush('error', { status: 500, statusText: 'Server Error' });

    comp.activeTab.set('score');
    fixture.detectChanges();

    const alert = el.querySelector('[role="alert"]');
    expect(alert).withContext('score tab must render [role="alert"] on error').not.toBeNull();
    expect((alert as HTMLElement | null)?.textContent?.trim().length)
      .withContext('[role="alert"] text must not be blank')
      .toBeGreaterThan(0);
  });

  it('when POST /factors/score returns 422, panelErrors["score"] is set', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush(
      { detail: 'Unprocessable' },
      { status: 422, statusText: 'Unprocessable Entity' },
    );

    expect(comp.panelErrors()['score']).toBeTruthy();
  });

  // ── select: loading flag ─────────────────────────────────────────────────────

  it('when onRunSelect() is called, selectLoading() becomes true before response arrives', () => {
    comp.onRunSelect(SELECT_PAYLOAD);

    expect(comp.selectLoading())
      .withContext('selectLoading must be true while request is in-flight')
      .toBeTrue();

    http.expectOne(SELECT_URL).flush(SELECT_RESPONSE);
  });

  it('when POST /factors/select succeeds, selectLoading() becomes false', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush(SELECT_RESPONSE);

    expect(comp.selectLoading())
      .withContext('selectLoading must be false after successful response')
      .toBeFalse();
  });

  it('when onRunSelect() is called, panelErrors["select"] is cleared to null', () => {
    comp.panelErrors.update((p) => ({ ...p, select: 'prior error' }));
    comp.onRunSelect(SELECT_PAYLOAD);

    expect(comp.panelErrors()['select'])
      .withContext('panelErrors["select"] must be cleared on re-run')
      .toBeNull();

    http.expectOne(SELECT_URL).flush(SELECT_RESPONSE);
  });

  // ── select: success path ─────────────────────────────────────────────────────

  it('when POST /factors/select succeeds, selectResult() is populated', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush(SELECT_RESPONSE);

    expect(comp.selectResult()).not.toBeNull();
    expect(comp.selectResult()?.selected_tickers).toEqual(['AAPL', 'MSFT']);
    expect(comp.selectResult()?.count).toBe(2);
  });

  it('when POST /factors/select succeeds, selected_tickers contains ticker strings', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush(SELECT_RESPONSE);

    const result = comp.selectResult();
    expect(result?.selected_tickers.length).toBeGreaterThan(0);
    result?.selected_tickers.forEach((ticker) => {
      expect(typeof ticker).toBe('string');
      expect(ticker.length).toBeGreaterThan(0);
    });
  });

  it('when POST /factors/select succeeds, panelErrors["select"] stays null', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush(SELECT_RESPONSE);

    expect(comp.panelErrors()['select'] ?? null)
      .withContext('no error on success')
      .toBeNull();
  });

  it('when selectResult() is set, select tab renders app-select-panel', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush(SELECT_RESPONSE);

    comp.activeTab.set('select');
    fixture.detectChanges();

    expect(el.querySelector('app-select-panel'))
      .withContext('app-select-panel must be present on select tab')
      .not.toBeNull();
  });

  it('when POST /factors/select is called, HTTP method is POST', () => {
    comp.onRunSelect(SELECT_PAYLOAD);

    const req = http.expectOne(SELECT_URL);
    expect(req.request.method).toBe('POST');
    req.flush(SELECT_RESPONSE);
  });

  it('when POST /factors/select is called, request body contains the tickers array', () => {
    comp.onRunSelect(SELECT_PAYLOAD);

    const req = http.expectOne(SELECT_URL);
    expect((req.request.body as FactorSelectRequest).tickers).toEqual(SELECT_PAYLOAD.tickers);
    req.flush(SELECT_RESPONSE);
  });

  // ── select: error path ───────────────────────────────────────────────────────

  it('when POST /factors/select returns 500, panelErrors["select"] is set', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.panelErrors()['select'])
      .withContext('panelErrors["select"] must be truthy on 500')
      .toBeTruthy();
  });

  it('when POST /factors/select errors, selectLoading() becomes false', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.selectLoading())
      .withContext('selectLoading must be false after error')
      .toBeFalse();
  });

  it('when POST /factors/select errors, selectResult() stays null', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.selectResult()).toBeNull();
  });

  it('when POST /factors/select errors, select tab shows [role="alert"] banner', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush('error', { status: 500, statusText: 'Server Error' });

    comp.activeTab.set('select');
    fixture.detectChanges();

    const alert = el.querySelector('[role="alert"]');
    expect(alert).withContext('select tab must render [role="alert"] on error').not.toBeNull();
    expect((alert as HTMLElement | null)?.textContent?.trim().length)
      .withContext('[role="alert"] text must not be blank')
      .toBeGreaterThan(0);
  });

  it('when POST /factors/select returns 404, panelErrors["select"] is set', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush(
      { detail: 'Not Found' },
      { status: 404, statusText: 'Not Found' },
    );

    expect(comp.panelErrors()['select']).toBeTruthy();
  });

  // ── mutual independence ───────────────────────────────────────────────────────

  it('score error does not pollute panelErrors["select"]', () => {
    comp.onRunScore(SCORE_PAYLOAD);
    http.expectOne(SCORE_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.panelErrors()['select'])
      .withContext('select error key must not be set by score failure')
      .toBeFalsy();
  });

  it('select error does not pollute panelErrors["score"]', () => {
    comp.onRunSelect(SELECT_PAYLOAD);
    http.expectOne(SELECT_URL).flush('error', { status: 500, statusText: 'Server Error' });

    expect(comp.panelErrors()['score'])
      .withContext('score error key must not be set by select failure')
      .toBeFalsy();
  });
});
