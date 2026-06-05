import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeRebalancePreview } from '../../../testing';
import { TradePreviewPanelComponent } from './trade-preview-panel';

describe('TradePreviewPanelComponent', () => {
  let fixture: ComponentFixture<TradePreviewPanelComponent>;
  let comp: TradePreviewPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [TradePreviewPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(TradePreviewPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when preview is null, the KPIs show dash/zero placeholders', () => {
    expect(comp.trades()).toEqual([]);
    expect(comp.totalTurnover()).toBe(0);
    expect(comp.kpiPortfolioValue()).toBe('—');
  });

  it('when populated, totalTurnover is half the summed absolute deltas', () => {
    fixture.componentRef.setInput('preview', makeRebalancePreview()); // one trade Δ -0.05
    expect(comp.totalTurnover()).toBeCloseTo(0.025); // |−0.05| / 2
    expect(comp.kpiTurnover()).toBe('2.50%');
  });

  it('when populated, trade rows carry side and waterfall values are in bps', () => {
    fixture.componentRef.setInput('preview', makeRebalancePreview());
    expect(comp.tradeRows().length).toBe(1);
    expect(comp.tradeRows()[0]['side']).toBe('sell');
    expect(comp.waterfallValues()).toEqual([-500]); // -0.05 × 10000
  });
});
