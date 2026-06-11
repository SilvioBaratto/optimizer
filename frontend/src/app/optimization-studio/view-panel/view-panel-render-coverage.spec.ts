/**
 * view-panel-render-coverage.spec.ts — issue #942
 *
 * Template-branch coverage for ViewPanelComponent. Drives each framework `@if`
 * arm (black_litterman / opinion_pool / entropy_pooling) plus the per-framework
 * loading / error / result three-ways, and the run* methods (success + error +
 * empty-ticker guard). Logic-only assertions live in view-panel.spec.ts.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';
import { throwError } from 'rxjs';

import {
  configureTestBed,
  injectHttp,
  makeGenerateViewsResponse,
  makeOpinionPoolResponse,
  makeEntropyPoolingResponse,
} from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ViewPanelComponent } from './view-panel';
import { OptimizationService } from '../optimization.service';
import { environment } from '../../../environments/environment';

const API = environment.apiUrl;

describe('ViewPanelComponent — render-coverage (#942)', () => {
  let fixture: ComponentFixture<ViewPanelComponent>;
  let comp: ViewPanelComponent;
  let el: HTMLElement;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ imports: [ViewPanelComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ViewPanelComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    http = injectHttp();
    fixture.detectChanges();
  });

  afterEach(() => http.verify());

  function button(text: string): HTMLButtonElement | undefined {
    return Array.from(el.querySelectorAll('button')).find((b) => b.textContent?.includes(text)) as
      | HTMLButtonElement
      | undefined;
  }

  describe('framework selector arms', () => {
    it('when black_litterman is selected, the BL generator renders', () => {
      comp.setViewFramework('black_litterman');
      fixture.detectChanges();
      expect(el.textContent).toContain('Generate BL views');
    });

    it('when opinion_pool is selected, the opinion-pool runner renders', () => {
      comp.setViewFramework('opinion_pool');
      fixture.detectChanges();
      expect(el.textContent).toContain('Run Opinion Pool');
    });

    it('when entropy_pooling is selected, the entropy runner and date inputs render', () => {
      comp.setViewFramework('entropy_pooling');
      fixture.detectChanges();
      expect(el.textContent).toContain('Run Entropy Pooling');
      expect(el.querySelector('#viewStartDate')).not.toBeNull();
    });
  });

  describe('black-litterman three-way', () => {
    beforeEach(() => {
      comp.setViewFramework('black_litterman');
      fixture.detectChanges();
    });

    it('when loading, the button reads Generating and is disabled', () => {
      comp.blLoading.set(true);
      fixture.detectChanges();
      expect(button('Generating')?.disabled).toBe(true);
    });

    it('when an error is set, it is shown', () => {
      comp.blError.set('bl boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('bl boom');
    });

    it('when the response reports missing tickers, the warning is shown', () => {
      comp.blResponse.set(makeGenerateViewsResponse({ tickersMissingData: ['XYZ'] }));
      fixture.detectChanges();
      expect(el.textContent).toContain('Missing factor data');
    });

    it('when view rows exist, the data table renders', () => {
      comp.blResponse.set(makeGenerateViewsResponse());
      fixture.detectChanges();
      expect(el.querySelector('app-data-table')).not.toBeNull();
    });
  });

  describe('opinion-pool three-way', () => {
    beforeEach(() => {
      comp.setViewFramework('opinion_pool');
      fixture.detectChanges();
    });

    it('when loading, the button reads Running and is disabled', () => {
      comp.opinionLoading.set(true);
      fixture.detectChanges();
      expect(button('Running')?.disabled).toBe(true);
    });

    it('when an error is set, it is shown', () => {
      comp.opinionError.set('op boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('op boom');
    });

    it('when a response exists, persona weights and the data table render', () => {
      comp.opinionResponse.set(makeOpinionPoolResponse());
      fixture.detectChanges();
      expect(el.textContent).toContain('Persona weights');
      expect(el.querySelector('app-data-table')).not.toBeNull();
    });
  });

  describe('entropy-pooling three-way', () => {
    beforeEach(() => {
      comp.setViewFramework('entropy_pooling');
      fixture.detectChanges();
    });

    it('when no views are entered, the run button is disabled', () => {
      expect(button('Run entropy pooling')?.disabled).toBe(true);
    });

    it('when views are entered, the run button is enabled', () => {
      comp.meanViewsRaw.set('AAPL == 0.1');
      fixture.detectChanges();
      expect(button('Run entropy pooling')?.disabled).toBe(false);
    });

    it('when an error is set, it is shown', () => {
      comp.entropyError.set('ent boom');
      fixture.detectChanges();
      expect(el.textContent).toContain('ent boom');
    });

    it('when a response exists, the mu entries render', () => {
      comp.entropyResponse.set(makeEntropyPoolingResponse());
      fixture.detectChanges();
      expect(el.textContent).toContain('%');
    });
  });

  describe('run methods', () => {
    it('runBlackLitterman posts and stores the response', () => {
      comp.runBlackLitterman();
      http.expectOne(`${API}views/generate`).flush(makeGenerateViewsResponse());
      expect(comp.blResponse()).not.toBeNull();
      expect(comp.blLoading()).toBe(false);
    });

    it('runBlackLitterman is guarded when there are no tickers', () => {
      comp.tickersRaw.set('');
      comp.runBlackLitterman();
      expect(http.match(`${API}views/generate`).length).toBe(0);
    });

    it('runBlackLitterman records an error on failure', () => {
      comp.runBlackLitterman();
      http.expectOne(`${API}views/generate`).flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
      expect(comp.blError()).toBeTruthy();
      expect(comp.blLoading()).toBe(false);
    });

    it('runOpinionPool posts with two personas and stores the response', () => {
      comp.runOpinionPool();
      http.expectOne(`${API}views/opinion-pool`).flush(makeOpinionPoolResponse());
      expect(comp.opinionResponse()).not.toBeNull();
    });

    it('runOpinionPool records an error on failure', () => {
      comp.runOpinionPool();
      http.expectOne(`${API}views/opinion-pool`).flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
      expect(comp.opinionError()).toBeTruthy();
    });

    it('runEntropyPooling posts and stores the response', () => {
      comp.runEntropyPooling();
      http.expectOne(`${API}views/entropy-pooling`).flush(makeEntropyPoolingResponse());
      expect(comp.entropyResponse()).not.toBeNull();
    });

    it('runEntropyPooling is guarded when there are no tickers', () => {
      comp.tickersRaw.set('');
      comp.runEntropyPooling();
      expect(http.match(`${API}views/entropy-pooling`).length).toBe(0);
    });

    it('runEntropyPooling records an error on failure', () => {
      comp.runEntropyPooling();
      http.expectOne(`${API}views/entropy-pooling`).flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
      expect(comp.entropyError()).toBeTruthy();
    });
  });

  it('blViewRows formats negative and positive exposures, dropping near-zero weights', () => {
    comp.blResponse.set(makeGenerateViewsResponse({ p: [[-0.5, 0.5]], tickersWithData: ['AAPL', 'MSFT'] }));
    const exposure = comp.blViewRows()[0]['assetExposure'] as string;
    expect(exposure).toContain('-0.50·AAPL');
    expect(exposure).toContain('+0.50·MSFT');
  });

  // The error callbacks are typed (err: Error); an HttpErrorResponse always has
  // a `.message`, so the `?? '/views/... failed'` defaults are only reachable
  // when the source errors with a message-less value (spied service).
  describe('default error messages', () => {
    it('runBlackLitterman falls back to the default error on a message-less failure', () => {
      spyOn(TestBed.inject(OptimizationService), 'generateViews').and.returnValue(throwError(() => ({})));
      comp.runBlackLitterman();
      expect(comp.blError()).toBe('/views/generate failed');
    });

    it('runOpinionPool falls back to the default error on a message-less failure', () => {
      spyOn(TestBed.inject(OptimizationService), 'opinionPool').and.returnValue(throwError(() => ({})));
      comp.runOpinionPool();
      expect(comp.opinionError()).toBe('/views/opinion-pool failed');
    });

    it('runEntropyPooling falls back to the default error on a message-less failure', () => {
      spyOn(TestBed.inject(OptimizationService), 'entropyPooling').and.returnValue(throwError(() => ({})));
      comp.runEntropyPooling();
      expect(comp.entropyError()).toBe('/views/entropy-pooling failed');
    });
  });
});
