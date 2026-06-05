import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeConcentrationMetrics } from '../../../testing';
import { ConcentrationPanelComponent } from './concentration-panel';

describe('ConcentrationPanelComponent', () => {
  let fixture: ComponentFixture<ConcentrationPanelComponent>;
  let comp: ConcentrationPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [ConcentrationPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(ConcentrationPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when metrics are empty, hhi is zero and effective bets is the dash placeholder', () => {
    expect(comp.hhi()).toBe(0);
    expect(comp.effectiveBets()).toBe('--');
    expect(comp.top5Weight()).toBe('0.0%');
  });

  it('when populated, hhi, effective bets and top-5 weight derive from the weights', () => {
    fixture.componentRef.setInput('metrics', makeConcentrationMetrics()); // weight 0.6
    expect(comp.hhi()).toBeCloseTo(0.36);
    expect(comp.effectiveBets()).toBe('2.8'); // 1 / 0.36
    expect(comp.top5Weight()).toBe('60.0%');
  });
});
