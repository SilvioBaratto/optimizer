import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsGaugeComponent, type GaugeThreshold } from './echarts-gauge';

interface GaugeOption {
  series: Array<{
    type: string;
    data: Array<{ value: number }>;
    axisLine: { lineStyle: { color: Array<[number, string]> } };
  }>;
}

function build(comp: EchartsGaugeComponent, value: number, thresholds: GaugeThreshold[]): GaugeOption {
  return (
    comp as unknown as { buildOption(v: number, t: GaugeThreshold[]): GaugeOption }
  ).buildOption(value, thresholds);
}

describe('EchartsGaugeComponent', () => {
  let fixture: ComponentFixture<EchartsGaugeComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsGaugeComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsGaugeComponent);
    fixture.detectChanges();
  });

  it('when default, it mounts without throwing', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when buildOption runs, the gauge value derives from the input', () => {
    const opt = build(fixture.componentInstance, 42, []);
    expect(opt.series[0].type).toBe('gauge');
    expect(opt.series[0].data[0].value).toBe(42);
  });

  it('when no thresholds, the axis uses a single colour stop', () => {
    expect(build(fixture.componentInstance, 10, []).series[0].axisLine.lineStyle.color.length).toBe(1);
  });

  it('when a threshold is set, the axis colour stops extend to the threshold', () => {
    const thresholds: GaugeThreshold[] = [{ value: 50, color: '#ff0000' }];
    const colors = build(fixture.componentInstance, 10, thresholds).series[0].axisLine.lineStyle.color;
    expect(colors.length).toBe(2); // [0.5, red] + capped [1, red]
    expect(colors[0]).toEqual([0.5, '#ff0000']);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
