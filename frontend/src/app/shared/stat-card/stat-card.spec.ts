import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { StatCardComponent } from './stat-card';

describe('StatCardComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StatCardComponent],
      providers: [provideZonelessChangeDetection()],
    }).compileComponents();
  });

  function createCardWith(inputs: {
    delta?: number | null;
    deltaFormat?: 'percent' | 'absolute';
  }): StatCardComponent {
    const fixture = TestBed.createComponent(StatCardComponent);
    fixture.componentRef.setInput('label', 'Test');
    fixture.componentRef.setInput('value', '0');
    if (inputs.delta !== undefined) {
      fixture.componentRef.setInput('delta', inputs.delta);
    }
    if (inputs.deltaFormat !== undefined) {
      fixture.componentRef.setInput('deltaFormat', inputs.deltaFormat);
    }
    fixture.detectChanges();
    return fixture.componentInstance;
  }

  describe('deltaFormatted', () => {
    it('renders percent (default) with × 100 + % suffix for backwards compatibility', () => {
      const card = createCardWith({ delta: 0.0234 });
      expect(card.deltaFormatted()).toBe('+2.34%');
    });

    it('renders percent with negative sign for negative deltas', () => {
      const card = createCardWith({ delta: -0.0825, deltaFormat: 'percent' });
      expect(card.deltaFormatted()).toBe('-8.25%');
    });

    it('renders absolute deltas without × 100 scaling and without %', () => {
      const card = createCardWith({ delta: -0.12, deltaFormat: 'absolute' });
      expect(card.deltaFormatted()).toBe('-0.12');
    });

    it('renders absolute deltas with explicit + sign for positives', () => {
      const card = createCardWith({ delta: 1.5, deltaFormat: 'absolute' });
      expect(card.deltaFormatted()).toBe('+1.50');
    });

    it('returns empty string when delta is null', () => {
      const card = createCardWith({ delta: null });
      expect(card.deltaFormatted()).toBe('');
    });

    it('does not render Sharpe-sized absolute deltas as multi-thousand percentages', () => {
      // Regression for issue #432: a Sharpe ratio change of -12.64 must not
      // render as -1264.00%; it should be a plain absolute number.
      const card = createCardWith({ delta: -12.64, deltaFormat: 'absolute' });
      expect(card.deltaFormatted()).toBe('-12.64');
      expect(card.deltaFormatted()).not.toContain('%');
    });
  });
});
