import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { MomentPanelComponent } from './moment-panel';
import { environment } from '../../../environments/environment';

const ADAPT_URL = `${environment.apiUrl}llm-moments/adapt-factor-weights`;

describe('MomentPanelComponent', () => {
  let fixture: ComponentFixture<MomentPanelComponent>;
  let comp: MomentPanelComponent;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ imports: [MomentPanelComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(MomentPanelComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('when macro indicators are below 20 chars, no request fires and an error is set', () => {
    comp.macroIndicators.set('a'.repeat(19));
    comp.runAdaptFactorWeights();
    http.expectNone(ADAPT_URL);
    expect(comp.adaptError()).toContain('at least 20');
  });

  it('when macro indicators reach 20 chars, the adapt-factor-weights request fires', () => {
    comp.macroIndicators.set('a'.repeat(20));
    comp.runAdaptFactorWeights();
    const req = http.expectOne(ADAPT_URL);
    expect(req.request.method).toBe('POST');
    req.flush({ phase: 'expansion', weights: { value: 0.5 } });
    expect(comp.factorWeights()).toEqual({ value: 0.5 });
  });

  it('when the request errors, calibrateDeltaError-style error is surfaced', () => {
    comp.macroIndicators.set('a'.repeat(25));
    comp.runAdaptFactorWeights();
    http.expectOne(ADAPT_URL).flush({ detail: 'llm down' }, { status: 502, statusText: 'Bad Gateway' });
    expect(comp.adaptError()).toBeTruthy();
  });
});
