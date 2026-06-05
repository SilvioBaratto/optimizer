import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsWaterfallComponent } from './echarts-waterfall';

interface WaterfallOption {
  series: Array<{
    data: Array<{ value: number }>;
    markLine?: { data: Array<{ yAxis: number }> };
  }>;
  xAxis: { data: string[] };
}

function build(comp: EchartsWaterfallComponent, categories: string[], values: number[]): WaterfallOption {
  return (
    comp as unknown as { buildOption(c: string[], v: number[]): WaterfallOption }
  ).buildOption(categories, values);
}

describe('EchartsWaterfallComponent', () => {
  let fixture: ComponentFixture<EchartsWaterfallComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsWaterfallComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsWaterfallComponent);
    fixture.detectChanges();
  });

  it('when empty, it mounts without throwing', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when mixed values with a non-zero base, the stacked bars and base derive from input', () => {
    fixture.componentRef.setInput('baseValue', 100);
    fixture.detectChanges();
    const opt = build(fixture.componentInstance, ['Start', 'Gain', 'Loss'], [10, 20, -5]);
    expect(opt.xAxis.data).toEqual(['Start', 'Gain', 'Loss']);
    expect(opt.series.length).toBe(2); // transparent base + visible bars
    // visible bar magnitudes use absolute deltas
    expect(opt.series[1].data.map((d) => d.value)).toEqual([10, 20, 5]);
    // base markLine anchors at the baseValue
    expect(opt.series[1].markLine?.data[0].yAxis).toBe(100);
  });

  it('when categories are empty, no visible bars are produced', () => {
    expect(build(fixture.componentInstance, [], []).series[1].data.length).toBe(0);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
