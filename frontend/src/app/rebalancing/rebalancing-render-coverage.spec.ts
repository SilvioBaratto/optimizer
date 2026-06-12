/**
 * rebalancing-render-coverage.spec.ts — issue #940
 *
 * Template-branch coverage for RebalancingComponent. Mounts the full template
 * and drives the three-way gate, 5 tab arms, KPI ternaries and the
 * pending-activate confirmation overlay by setting STATE signals.
 *
 * Portfolio selection is now driven by PortfolioContextService (global context),
 * not a local PortfolioApiService call — no HTTP bootstrap is needed.
 */

import { provideZonelessChangeDetection, signal, computed } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed, ComponentFixture } from '@angular/core/testing';

import {
  installResizeObserverStub,
  makeDriftResponse,
  makeRebalanceDecideResponse,
  makeRebalancingPolicyDto,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { RebalancingComponent } from './rebalancing';
import { PortfolioContextService } from '../core/services/portfolio-context.service';

function makeCtxMock(name: string | null = 'Test Portfolio', id: string | null = 'test-id') {
  const nameSig = signal(name);
  const idSig   = signal(id);
  return {
    currentPortfolioId:   idSig,
    currentPortfolioName: computed(() => nameSig()),
    hasPortfolio:         computed(() => idSig() !== null),
    selectedPortfolio:    computed(() => (idSig() ? { id: idSig()!, name: nameSig() ?? '' } : null)),
    dateRange:            signal({ preset: '1Y' as const, start: new Date('2024-01-01'), end: new Date('2025-01-01') }),
    benchmark:            signal('SPY'),
    activeMode:           signal('backtest' as const),
    isLive:               computed(() => false),
    isBacktest:           computed(() => true),
    isPaper:              computed(() => false),
    dateRangeLabel:       computed(() => '1Y'),
    dateRangeDays:        computed(() => 365),
    setPortfolio:   jasmine.createSpy('setPortfolio'),
    setMode:        jasmine.createSpy('setMode'),
    setPreset:      jasmine.createSpy('setPreset'),
    setCustomRange: jasmine.createSpy('setCustomRange'),
    setBenchmark:   jasmine.createSpy('setBenchmark'),
    reset:          jasmine.createSpy('reset'),
  };
}

describe('RebalancingComponent — render-coverage (issue #940)', () => {
  let fixture: ComponentFixture<RebalancingComponent>;
  let comp: RebalancingComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();
    await TestBed.configureTestingModule({
      imports: [RebalancingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtxMock() },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(RebalancingComponent);
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

    it('when hasError is true, the error banner is shown with the message', () => {
      comp.isLoading.set(false);
      comp.hasError.set(true);
      comp.errorMessage.set('rebal boom');
      fixture.detectChanges();
      const alertEl = el.querySelector('[role="alert"]') ?? el.querySelector('app-page-error-banner');
      expect(alertEl).not.toBeNull();
      expect(el.textContent).toContain('rebal boom');
    });

    it('when not loading and no error, the tab group is rendered', () => {
      comp.isLoading.set(false);
      fixture.detectChanges();
      expect(el.querySelector('app-tab-group')).not.toBeNull();
    });
  });

  describe('tab arms', () => {
    const cases = [
      { tab: 'status', selector: 'app-status-panel' },
      { tab: 'policy', selector: 'app-policy-panel' },
      { tab: 'trade-preview', selector: 'app-trade-preview-panel' },
      { tab: 'history', selector: 'app-history-panel' },
      { tab: 'what-if', selector: 'app-whatif-panel' },
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

  describe('KPI cards', () => {
    it('when there is no drift/decide/policy data, KPI cards show em dashes', () => {
      comp.isLoading.set(false);
      fixture.detectChanges();
      expect(cardText('Max Drift')).toContain('—');
      expect(cardText('Active Policy')).toContain('—');
      expect(cardText('Est. Cost')).toContain('—');
      expect(cardText('Breached Assets')).toContain('0');
    });

    it('when drift, decide and an active policy are set, KPI cards show values', () => {
      comp.isLoading.set(false);
      comp.driftResponse.set(makeDriftResponse({ breachedCount: 2 }));
      comp.decideResponse.set(makeRebalanceDecideResponse());
      comp.policies.set([makeRebalancingPolicyDto({ isActive: true, name: 'Quarterly Rule' })]);
      fixture.detectChanges();
      expect(cardText('Max Drift')).not.toContain('—');
      expect(cardText('Active Policy')).toContain('Quarterly Rule');
      expect(cardText('Est. Cost')).not.toContain('—');
      expect(cardText('Breached Assets')).toContain('2');
    });
  });

  describe('pending-activate confirmation overlay', () => {
    it('when pendingActivateId is null, no confirmation dialog is shown', () => {
      comp.isLoading.set(false);
      comp.pendingActivateId.set(null);
      fixture.detectChanges();
      expect(el.querySelector('[role="dialog"]')).toBeNull();
    });

    it('when pendingActivateId is set, the confirmation dialog is shown', () => {
      comp.isLoading.set(false);
      comp.pendingActivateId.set('pol-1');
      fixture.detectChanges();
      expect(el.querySelector('[role="dialog"]')).not.toBeNull();
      expect(el.querySelector('[data-testid="confirm-activate"]')).not.toBeNull();
      expect(el.textContent).toContain('Activate this policy?');
    });
  });
});
