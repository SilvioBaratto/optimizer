import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { AiControlRoomComponent } from './ai-control-room';
import { FEATURE_AI_DECISION_LOG_TOKEN } from '../../config/feature-flags';
import { ICON_PROVIDER } from '../../icons';

function createComponent(flagValue: boolean) {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      provideRouter([]),
      ICON_PROVIDER,
      { provide: FEATURE_AI_DECISION_LOG_TOKEN, useValue: flagValue },
    ],
  });
  const fx = TestBed.createComponent(AiControlRoomComponent);
  fx.detectChanges();
  return fx;
}

describe('AiControlRoomComponent — feature flag gating', () => {
  afterEach(() => TestBed.resetTestingModule());

  describe('when FEATURE_AI_DECISION_LOG is OFF (default)', () => {
    it('exposes only the pipeline tab', () => {
      const fx = createComponent(false);
      const tabs = fx.componentInstance.tabs();
      expect(tabs.length).toBe(1);
      expect(tabs[0].id).toBe('pipeline');
    });

    it("defaults activeTab to 'pipeline'", () => {
      const fx = createComponent(false);
      expect(fx.componentInstance.activeTab()).toBe('pipeline');
    });

    it('does not render the decision-log KPI row', () => {
      const fx = createComponent(false);
      expect(fx.componentInstance.decisionLogEnabled).toBe(false);
      const cards: NodeListOf<Element> =
        fx.nativeElement.querySelectorAll('app-stat-card');
      expect(cards.length).toBe(0);
    });
  });

  describe('when FEATURE_AI_DECISION_LOG is ON', () => {
    it('exposes all four tabs in the documented order', () => {
      const fx = createComponent(true);
      const ids = fx.componentInstance.tabs().map((t) => t.id);
      expect(ids).toEqual(['overview', 'history', 'agents', 'pipeline']);
    });

    it("defaults activeTab to 'overview'", () => {
      const fx = createComponent(true);
      expect(fx.componentInstance.activeTab()).toBe('overview');
    });

    it('renders the decision-log KPI row when the overview tab is active', () => {
      const fx = createComponent(true);
      expect(fx.componentInstance.decisionLogEnabled).toBe(true);
      const cards = fx.nativeElement.querySelectorAll('app-stat-card');
      expect(cards.length).toBeGreaterThan(0);
    });
  });
});
