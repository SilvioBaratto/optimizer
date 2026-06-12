/**
 * attribution-render-coverage.spec.ts — issue #940
 *
 * Template-branch coverage for AttributionComponent. Mounts the full template
 * and drives the three-way gate, portfolio select, 4 tab arms, form-validity
 * disabled states, loading button text, error text, the unclassified-benchmark
 * warning and KPI ternaries by setting STATE signals directly.
 *
 * PortfolioContextService is mocked with a null portfolio so the context-wiring
 * effect does not fire HTTP calls during template-branch testing.
 */

import { provideZonelessChangeDetection, signal, computed, NO_ERRORS_SCHEMA } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { of } from 'rxjs';

import {
  installResizeObserverStub,
  makePortfolioDto,
  makeBrinsonResponse,
  makeFactorAttributionResponse,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { AttributionService } from './attribution.service';

// Null-portfolio context prevents the effect from triggering service calls
// during template-branch tests so we can set signals directly.
function makeNullCtxMock() {
  return {
    currentPortfolioId: signal<string | null>(null),
    currentPortfolioName: computed(() => null),
    selectedPortfolio: computed(() => null),
    dateRange: signal({ preset: '1Y' as const, start: new Date('2024-01-01'), end: new Date('2025-01-01') }),
    benchmark: signal('SPY'),
    hasPortfolio: computed(() => false),
    activeMode: signal('backtest' as const),
    isLive: computed(() => false),
    isBacktest: computed(() => true),
    isPaper: computed(() => false),
    dateRangeLabel: computed(() => '1Y'),
    dateRangeDays: computed(() => 365),
    setPortfolio: jasmine.createSpy('setPortfolio'),
    setMode: jasmine.createSpy('setMode'),
    setPreset: jasmine.createSpy('setPreset'),
    setCustomRange: jasmine.createSpy('setCustomRange'),
    setBenchmark: jasmine.createSpy('setBenchmark'),
    reset: jasmine.createSpy('reset'),
  };
}

function makeAttrSvcStub() {
  return {
    getBrinsonAttribution: jasmine.createSpy('getBrinsonAttribution').and.returnValue(of(null)),
    getFactorAttribution: jasmine.createSpy('getFactorAttribution').and.returnValue(of(null)),
    loadAttributionData: jasmine.createSpy('loadAttributionData').and.returnValue(of({})),
    getAttribution: jasmine.createSpy('getAttribution').and.returnValue(of({})),
    brinson: jasmine.createSpy('brinson').and.returnValue(of(makeBrinsonResponse())),
    factor: jasmine.createSpy('factor').and.returnValue(of(makeFactorAttributionResponse())),
  };
}

describe('AttributionComponent — render-coverage (issue #940)', () => {
  let fixture: ComponentFixture<AttributionComponent>;
  let comp: AttributionComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();
    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeNullCtxMock() },
        { provide: AttributionService, useValue: makeAttrSvcStub() },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
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

  function button(text: string): HTMLButtonElement | undefined {
    return Array.from(el.querySelectorAll('button')).find((b) =>
      b.textContent?.includes(text),
    ) as HTMLButtonElement | undefined;
  }

  describe('loading / error / success three-way', () => {
    it('when isLoading is true, skeleton tiles are shown', () => {
      comp.isLoading.set(true);
      fixture.detectChanges();
      expect(el.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
    });

    it('when hasError is true, the error block is shown', () => {
      comp.isLoading.set(false);
      comp.hasError.set(true);
      comp.errorMessage.set('attr boom');
      fixture.detectChanges();
      const alertEl =
        el.querySelector('[role="alert"]') ??
        el.querySelector('app-page-error-banner');
      expect(alertEl).not.toBeNull();
    });

    it('when not loading and no error, the tab group is rendered', () => {
      comp.isLoading.set(false);
      fixture.detectChanges();
      expect(el.querySelector('app-tab-group')).not.toBeNull();
    });
  });

  describe('portfolio selector @if + @for', () => {
    it('when there are no portfolios, the selector is absent', () => {
      comp.isLoading.set(false);
      comp.portfolios.set([]);
      fixture.detectChanges();
      expect(el.querySelector('select[aria-label="Portfolio"]')).toBeNull();
    });

    it('when there are two portfolios, the selector renders two options', () => {
      comp.isLoading.set(false);
      comp.portfolios.set([
        makePortfolioDto({ id: 'a', name: 'Alpha' }),
        makePortfolioDto({ id: 'b', name: 'Beta' }),
      ]);
      fixture.detectChanges();
      const select = el.querySelector<HTMLSelectElement>('select[aria-label="Portfolio"]');
      expect(select?.options.length).toBe(2);
    });
  });

  describe('tab arms', () => {
    const cases = [
      { tab: 'brinson', selector: 'app-brinson-panel' },
      { tab: 'multi-level', selector: 'app-saa-taa-panel' },
      { tab: 'factor', selector: 'app-factor-attribution-panel' },
      { tab: 'holdings', selector: 'app-holdings-attribution-panel' },
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

  describe('form validity gating', () => {
    it('when the form is invalid, Run Brinson is disabled and a warning is shown', () => {
      comp.isLoading.set(false);
      comp.portfolioWeights.set({}); // empty → sumsToOne false → invalid
      fixture.detectChanges();
      expect(button('Run Brinson')?.disabled).toBe(true);
      expect(el.textContent).toContain('must be non-empty and each sum to 1');
    });

    it('when the form is valid, Run Brinson is enabled and the warning is hidden', () => {
      comp.isLoading.set(false);
      comp.portfolioWeights.set({ AAPL: 1 }); // sums to 1; benchmark SPY:1.0 default
      fixture.detectChanges();
      expect(button('Run Brinson')?.disabled).toBe(false);
      expect(el.textContent).not.toContain('must be non-empty and each sum to 1');
    });
  });

  describe('loading button states', () => {
    it('when brinsonLoading is true, the Brinson button reads Running and is disabled', () => {
      comp.isLoading.set(false);
      comp.brinsonLoading.set(true);
      fixture.detectChanges();
      const btn = button('Running');
      expect(btn).not.toBeUndefined();
      expect(btn?.disabled).toBe(true);
    });

    it('when factorLoading is true, the Factor button is disabled', () => {
      comp.isLoading.set(false);
      comp.factorLoading.set(true);
      fixture.detectChanges();
      expect(button('Running')?.disabled).toBe(true);
    });
  });

  describe('error text and KPI cards', () => {
    it('when brinsonError/factorError are set, both error lines are shown', () => {
      comp.isLoading.set(false);
      comp.brinsonError.set('brinson failed');
      comp.factorError.set('factor failed');
      fixture.detectChanges();
      expect(el.textContent).toContain('Brinson: brinson failed');
      expect(el.textContent).toContain('Factor: factor failed');
    });

    it('when responses are null, KPI cards show em dashes', () => {
      comp.isLoading.set(false);
      comp.brinsonResponse.set(null);
      comp.factorResponse.set(null);
      fixture.detectChanges();
      expect(cardText('Total Active Return')).toContain('—');
      expect(cardText('Factor Residual')).toContain('—');
    });

    it('when responses are set, KPI cards show computed values', () => {
      comp.isLoading.set(false);
      comp.brinsonResponse.set(makeBrinsonResponse());
      comp.factorResponse.set(makeFactorAttributionResponse());
      fixture.detectChanges();
      expect(cardText('Total Active Return')).not.toContain('—');
      expect(cardText('Factor Residual')).not.toContain('—');
    });
  });

  describe('unclassified-benchmark warning', () => {
    it('when the benchmark is fully classified, no warning banner is shown', () => {
      comp.isLoading.set(false);
      comp.brinsonResponse.set(makeBrinsonResponse());
      fixture.detectChanges();
      expect(el.textContent).not.toContain('resolves to a single');
    });

    it('when the benchmark is entirely Unclassified, the warning banner is shown', () => {
      comp.isLoading.set(false);
      comp.brinsonResponse.set(
        makeBrinsonResponse({
          sectors: [
            {
              sector: 'Unclassified',
              portfolioWeight: 1,
              benchmarkWeight: 1,
              portfolioReturn: 0,
              benchmarkReturn: 0,
              allocationEffect: 0,
              selectionEffect: 0,
              interactionEffect: 0,
              totalEffect: 0,
            },
          ],
        }),
      );
      fixture.detectChanges();
      expect(el.textContent).toContain('resolves to a single');
    });
  });
});
