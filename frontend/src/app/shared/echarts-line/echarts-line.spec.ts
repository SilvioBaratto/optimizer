import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsLineComponent } from './echarts-line';

interface LineOption {
  series: Array<{ data: number[]; areaStyle?: unknown }>;
  xAxis: { data: string[] };
}

function build(comp: EchartsLineComponent, labels: string[], values: number[]): LineOption {
  return (comp as unknown as { buildOption(l: string[], v: number[]): LineOption }).buildOption(labels, values);
}

describe('EchartsLineComponent', () => {
  let fixture: ComponentFixture<EchartsLineComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsLineComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsLineComponent);
  });

  it('when inputs are empty, it mounts without throwing', () => {
    expect(() => fixture.detectChanges()).not.toThrow();
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when populated, the series and axis derive from labels/values', () => {
    fixture.componentRef.setInput('labels', ['Jan', 'Feb']);
    fixture.componentRef.setInput('values', [1, 2]);
    fixture.detectChanges();
    const opt = build(fixture.componentInstance, ['Jan', 'Feb'], [1, 2]);
    expect(opt.xAxis.data).toEqual(['Jan', 'Feb']);
    expect(opt.series[0].data).toEqual([1, 2]);
  });

  it('when areaFill is off, the series has no areaStyle', () => {
    fixture.detectChanges();
    expect(build(fixture.componentInstance, ['a'], [1]).series[0].areaStyle).toBeUndefined();
  });

  it('when areaFill is on, the series gains an areaStyle', () => {
    fixture.componentRef.setInput('areaFill', true);
    fixture.detectChanges();
    expect(build(fixture.componentInstance, ['a'], [1]).series[0].areaStyle).toBeDefined();
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
