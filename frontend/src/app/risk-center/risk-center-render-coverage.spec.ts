/**
 * risk-center-render-coverage.spec.ts — issue #940
 *
 * Template-branch coverage for RiskCenterComponent. Mounts the full template
 * and drives each @if / @else / @for arm (three-way gate, 7 tab arms,
 * KPI ternaries, breach badge) by setting STATE signals directly.
 * HTTP from the constructor's effect-driven analytics is left pending
 * (no http.verify() in afterEach — requests left open are acceptable here).
 */

import { provideZonelessChangeDetection, signal } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed, ComponentFixture } from '@angular/core/testing';

import { installResizeObserverStub, makeVarApiResponse } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { RiskCenterComponent } from './risk-center';
import { PortfolioContextService } from '../core/services/portfolio-context.service';

function makeCtxStub(name: string | null = 'Test Portfolio', id: string | null = 'test-id') {
  return {
    currentPortfolioId: signal<string | null>(id),
    currentPortfolioName: signal<string | null>(name),
  };
}

describe('RiskCenterComponent — render-coverage (issue #940)', () => {
  let fixture: ComponentFixture<RiskCenterComponent>;
  let comp: RiskCenterComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();
    await TestBed.configureTestingModule({
      imports: [RiskCenterComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtxStub() },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(RiskCenterComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  function cardText(label: string): string {
    const card = Array.from(el.querySelectorAll('app-stat-card')).find((c) =>
      c.textContent?.includes(label),
    );
    return card?.textContent ?? '';
  }

  describe('loading / error / success three-way', () => {
    it('when isLoading is true, skeleton tiles are shown', () => {
      comp.isLoading.set(true);
      fixture.detectChanges();
      expect(el.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
    });

    it('when hasError is true, the page-error-banner is shown', () => {
      comp.isLoading.set(false);
      comp.hasError.set(true);
      comp.errorMessage.set('risk boom');
      fixture.detectChanges();
      const banner = el.querySelector('app-page-error-banner');
      expect(banner).not.toBeNull();
    });

    it('when not loading and no error, the tab group is rendered', () => {
      comp.isLoading.set(false);
      fixture.detectChanges();
      expect(el.querySelector('app-tab-group')).not.toBeNull();
      expect(el.querySelectorAll('.skeleton').length).toBe(0);
    });
  });

  describe('tab arms', () => {
    const cases = [
      { tab: 'var', selector: 'app-var-panel' },
      { tab: 'stress', selector: 'app-stress-panel' },
      { tab: 'correlation', selector: 'app-correlation-panel' },
      { tab: 'factor', selector: 'app-factor-panel' },
      { tab: 'concentration', selector: 'app-concentration-panel' },
      { tab: 'liquidity', selector: 'app-liquidity-panel' },
      { tab: 'limits', selector: 'app-limits-panel' },
    ];

    cases.forEach(({ tab, selector }) => {
      it(`when activeTab is "${tab}", ${selector} is rendered`, () => {
        comp.isLoading.set(false);
        comp.activeTab.set(tab);
        fixture.detectChanges();
        expect(el.querySelector(selector)).not.toBeNull();
      });
    });
  });

  describe('panel error banners', () => {
    it('when a panel error is set for var, alert-banner is rendered in the var tab', () => {
      comp.isLoading.set(false);
      comp.activeTab.set('var');
      comp.panelErrors.set({ var: 'VaR load failed' });
      fixture.detectChanges();
      const banner = el.querySelector('app-alert-banner');
      expect(banner).not.toBeNull();
    });

    it('when a panel error is set for limits, alert-banner is rendered in the limits tab', () => {
      comp.isLoading.set(false);
      comp.activeTab.set('limits');
      comp.panelErrors.set({ limits: 'Limits load failed' });
      fixture.detectChanges();
      const banner = el.querySelector('app-alert-banner');
      expect(banner).not.toBeNull();
    });
  });

  describe('KPI cards reflect varResponse', () => {
    it('when varResponse is null, the VaR 95% card shows an em dash', () => {
      comp.isLoading.set(false);
      comp.varResponse.set(null);
      fixture.detectChanges();
      expect(cardText('VaR 95%')).toContain('—');
    });

    it('when varResponse is set, the VaR 95% card shows a computed value', () => {
      comp.isLoading.set(false);
      comp.varResponse.set(makeVarApiResponse());
      fixture.detectChanges();
      expect(cardText('VaR 95%')).not.toContain('—');
    });
  });

  describe('breach count badge', () => {
    it('when breachCount is 0, the Breach count card shows 0', () => {
      comp.isLoading.set(false);
      comp.limitsResponse.set({ items: [], breachCount: 0 });
      fixture.detectChanges();
      expect(cardText('Breach count')).toContain('0');
    });

    it('when breachCount is positive, the Breach count card shows the count', () => {
      comp.isLoading.set(false);
      comp.limitsResponse.set({ items: [], breachCount: 3 });
      fixture.detectChanges();
      expect(cardText('Breach count')).toContain('3');
    });
  });
});
