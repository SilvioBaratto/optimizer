/**
 * Issue #1036 — PortfolioContextService defaults activeMode() to 'live'
 * and reset() restores 'live'. setMode / isLive / isBacktest / isPaper
 * must remain on the service.
 * Source-blind: derived from acceptance criteria only.
 */
import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';

import { PortfolioContextService } from './portfolio-context.service';

describe('PortfolioContextService — live mode default (#1036)', () => {
  let svc: PortfolioContextService;
  let http: HttpTestingController;

  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });

    svc = TestBed.inject(PortfolioContextService);
    http = TestBed.inject(HttpTestingController);
    // Flush any auto-triggered HTTP call from the constructor
    http.match(() => true).forEach(req => req.flush({ items: [] }));
  });

  afterEach(() => {
    http.match(() => true).forEach(req => req.flush({ items: [] }));
    localStorage.clear();
    http.verify();
  });

  it('when service is created, activeMode() returns "live"', () => {
    expect(svc.activeMode()).toBe('live');
  });

  it('when mode is changed to backtest then reset() is called, activeMode() returns "live"', () => {
    svc.setMode('backtest');
    svc.reset();
    expect(svc.activeMode()).toBe('live');
  });

  it('when mode is changed to paper then reset() is called, activeMode() returns "live"', () => {
    svc.setMode('paper');
    svc.reset();
    expect(svc.activeMode()).toBe('live');
  });

  it('when service is created, isLive() returns true', () => {
    expect(svc.isLive()).toBeTrue();
  });

  it('when setMode("backtest") is called, isBacktest() returns true', () => {
    svc.setMode('backtest');
    expect(svc.isBacktest()).toBeTrue();
  });

  it('when setMode("paper") is called, isPaper() returns true', () => {
    svc.setMode('paper');
    expect(svc.isPaper()).toBeTrue();
  });

  it('when setMode("live") is called after backtest, isLive() returns true', () => {
    svc.setMode('backtest');
    svc.setMode('live');
    expect(svc.isLive()).toBeTrue();
  });

  it('when service is inspected, setMode method exists', () => {
    expect(typeof svc.setMode).toBe('function');
  });

  it('when service is inspected, isLive computed exists', () => {
    expect(svc.isLive).toBeDefined();
  });

  it('when service is inspected, isBacktest computed exists', () => {
    expect(svc.isBacktest).toBeDefined();
  });

  it('when service is inspected, isPaper computed exists', () => {
    expect(svc.isPaper).toBeDefined();
  });
});
