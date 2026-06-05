import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsCalendarHeatmapComponent } from './echarts-calendar-heatmap';

interface CalendarOption {
  series: Array<{ data: unknown[] }>;
  xAxis: { data: string[] };
  yAxis: { data: string[] };
}

function build(
  comp: EchartsCalendarHeatmapComponent,
  years: string[],
  months: string[],
  data: number[][],
): CalendarOption {
  return (
    comp as unknown as { buildOption(y: string[], m: string[], d: number[][]): CalendarOption }
  ).buildOption(years, months, data);
}

describe('EchartsCalendarHeatmapComponent', () => {
  let fixture: ComponentFixture<EchartsCalendarHeatmapComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsCalendarHeatmapComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsCalendarHeatmapComponent);
    fixture.detectChanges();
  });

  it('when empty, it mounts without throwing', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when populated, the axes use months/years and the grid is rows×cols', () => {
    const opt = build(fixture.componentInstance, ['2025', '2026'], ['Jan', 'Feb', 'Mar'], [
      [0.01, -0.02, 0.03],
      [0.04, 0.05, -0.06],
    ]);
    expect(opt.xAxis.data).toEqual(['Jan', 'Feb', 'Mar']);
    expect(opt.yAxis.data).toEqual(['2025', '2026']);
    expect(opt.series[0].data.length).toBe(6); // 2 years × 3 months
  });

  it('when years/months are empty, no cells are produced', () => {
    expect(build(fixture.componentInstance, [], [], []).series[0].data.length).toBe(0);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
