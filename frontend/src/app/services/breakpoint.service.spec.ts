import { TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../testing';
import { BreakpointService } from './breakpoint.service';

describe('BreakpointService', () => {
  let svc: BreakpointService;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ withHttp: false });
    svc = TestBed.inject(BreakpointService);
  });

  it('when injected twice, the providedIn root singleton resolves once', () => {
    expect(TestBed.inject(BreakpointService)).toBe(svc);
  });

  it('when neither mobile nor tablet, chartHeight is the desktop value', () => {
    svc.isMobile.set(false);
    svc.isTablet.set(false);
    expect(svc.chartHeight()).toBe(380);
  });

  it('when tablet, chartHeight is the tablet value', () => {
    svc.isMobile.set(false);
    svc.isTablet.set(true);
    expect(svc.chartHeight()).toBe(280);
  });

  it('when mobile, chartHeight is the mobile value', () => {
    svc.isMobile.set(true);
    expect(svc.chartHeight()).toBe(220);
  });

  it('when mobile takes precedence over tablet, the mobile value wins', () => {
    svc.isMobile.set(true);
    svc.isTablet.set(true);
    expect(svc.chartHeight()).toBe(220);
  });
});
