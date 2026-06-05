import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsHeatmapComponent } from './echarts-heatmap';

interface HeatmapOption {
  series: Array<{ data: unknown[] }>;
  xAxis: { data: string[] };
  yAxis: { data: string[] };
}

function build(comp: EchartsHeatmapComponent, assets: string[], matrix: number[][]): HeatmapOption {
  return (comp as unknown as { buildOption(a: string[], m: number[][]): HeatmapOption }).buildOption(assets, matrix);
}

describe('EchartsHeatmapComponent', () => {
  let fixture: ComponentFixture<EchartsHeatmapComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsHeatmapComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsHeatmapComponent);
    fixture.detectChanges(); // resolve required viewChild(container) before buildOption
  });

  it('when empty, it mounts without throwing', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when populated, the axes use the assets and the grid is N×N', () => {
    const opt = build(fixture.componentInstance, ['AAPL', 'MSFT'], [
      [1, 0.5],
      [0.5, 1],
    ]);
    expect(opt.xAxis.data).toEqual(['AAPL', 'MSFT']);
    expect(opt.yAxis.data).toEqual(['AAPL', 'MSFT']);
    expect(opt.series[0].data.length).toBe(4); // 2 × 2 cells
  });

  it('when populated with empty assets, no cells are produced', () => {
    expect(build(fixture.componentInstance, [], []).series[0].data.length).toBe(0);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
