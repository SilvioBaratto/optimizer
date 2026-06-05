import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeLiquidityMetrics } from '../../../testing';
import { LiquidityPanelComponent } from './liquidity-panel';

describe('LiquidityPanelComponent', () => {
  let fixture: ComponentFixture<LiquidityPanelComponent>;
  let comp: LiquidityPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [LiquidityPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(LiquidityPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when metrics are empty, totals are zeroed and weighted-average is the dash', () => {
    expect(comp.totalCost()).toBe('0 bps');
    expect(comp.maxDays()).toBe('0.0d');
    expect(comp.weightedAvgDays()).toBe('--');
  });

  it('when populated, cost, max days and the bucket distribution derive from the metrics', () => {
    fixture.componentRef.setInput('metrics', makeLiquidityMetrics()); // 1.5d, cost 0.001, weight 0.6
    expect(comp.totalCost()).toBe('10 bps'); // 0.001 × 10000
    expect(comp.maxDays()).toBe('1.5d');
    expect(comp.weightedAvgDays()).toBe('1.50d');
    const buckets = comp.bucketDistribution();
    expect(buckets.length).toBe(5);
    expect(buckets[2].value).toBe(1); // 1.5d falls in the 1-2d bucket
  });
});
