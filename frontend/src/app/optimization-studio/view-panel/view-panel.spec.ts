import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ViewPanelComponent } from './view-panel';
import { environment } from '../../../environments/environment';

const OPINION_URL = `${environment.apiUrl}views/opinion-pool`;

describe('ViewPanelComponent', () => {
  let fixture: ComponentFixture<ViewPanelComponent>;
  let comp: ViewPanelComponent;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ imports: [ViewPanelComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ViewPanelComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('when only one persona is given, the opinion-pool request does not fire', () => {
    comp.personasRaw.set('VALUE_INVESTOR');
    comp.runOpinionPool();
    expect(http.match(OPINION_URL).length).toBe(0);
  });

  it('when two personas are given, the opinion-pool request fires', () => {
    comp.personasRaw.set('VALUE_INVESTOR, MOMENTUM_TRADER');
    comp.runOpinionPool();
    const req = http.expectOne(OPINION_URL);
    expect(req.request.method).toBe('POST');
    req.flush({
      nExperts: 0,
      tickers: [],
      tickersWithData: [],
      tickersMissingData: [],
      experts: [],
      icWeights: [],
      poolingType: 'linear',
      totalViews: 0,
    });
  });

  it('when no view text is entered, entropyHasViews is false', () => {
    expect(comp.entropyHasViews()).toBe(false);
  });

  it('when a mean view is entered, entropyHasViews becomes true', () => {
    comp.meanViewsRaw.set('AAPL > 0.05');
    expect(comp.entropyHasViews()).toBe(true);
  });
});
