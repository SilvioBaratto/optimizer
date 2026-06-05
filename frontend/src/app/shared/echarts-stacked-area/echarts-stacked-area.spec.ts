import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsStackedAreaComponent, type AreaSeries } from './echarts-stacked-area';

interface StackedOption {
  series: Array<{ name: string; data: number[] }>;
  xAxis: { data: string[] };
}

function build(comp: EchartsStackedAreaComponent, labels: string[], series: AreaSeries[]): StackedOption {
  return (
    comp as unknown as { buildOption(l: string[], s: AreaSeries[]): StackedOption }
  ).buildOption(labels, series);
}

const TWO_SERIES: AreaSeries[] = [
  { name: 'Equity', values: [1, 2, 3] },
  { name: 'Bonds', values: [4, 5, 6] },
];

describe('EchartsStackedAreaComponent', () => {
  let fixture: ComponentFixture<EchartsStackedAreaComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsStackedAreaComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsStackedAreaComponent);
    fixture.detectChanges(); // resolve required viewChild(container)
  });

  it('when empty, it mounts without throwing', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when series is empty, no series are produced', () => {
    expect(build(fixture.componentInstance, ['a', 'b', 'c'], []).series.length).toBe(0);
  });

  it('when two series are given, the axis and both stacked series derive from input', () => {
    const opt = build(fixture.componentInstance, ['a', 'b', 'c'], TWO_SERIES);
    expect(opt.xAxis.data).toEqual(['a', 'b', 'c']);
    expect(opt.series.length).toBe(2);
    expect(opt.series[0].name).toBe('Equity');
    expect(opt.series[1].data).toEqual([4, 5, 6]);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
