import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsBarComponent, type BarData } from './echarts-bar';

interface BarOption {
  series: Array<{ data: Array<{ value: number }> }>;
  xAxis: { data: string[] };
}

function build(comp: EchartsBarComponent, bars: BarData[]): BarOption {
  return (comp as unknown as { buildOption(b: BarData[]): BarOption }).buildOption(bars);
}

describe('EchartsBarComponent', () => {
  let fixture: ComponentFixture<EchartsBarComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsBarComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsBarComponent);
  });

  it('when data is empty, it mounts without throwing', () => {
    fixture.componentRef.setInput('data', []);
    expect(() => fixture.detectChanges()).not.toThrow();
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when data is populated, it mounts without throwing', () => {
    fixture.componentRef.setInput('data', [{ label: 'a', value: 0.5 }] as BarData[]);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('when buildOption runs, the series and axis derive from the data', () => {
    fixture.detectChanges();
    const opt = build(fixture.componentInstance, [
      { label: 'a', value: 0.5 },
      { label: 'b', value: -0.3 },
    ]);
    expect(opt.series[0].data.length).toBe(2);
    expect(opt.series[0].data[0].value).toBe(50); // 0.5 * 100
    expect(opt.xAxis.data.length).toBe(2);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
