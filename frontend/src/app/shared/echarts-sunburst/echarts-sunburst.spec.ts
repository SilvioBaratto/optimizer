import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsSunburstComponent, type SunburstNode } from './echarts-sunburst';

interface SunburstOption {
  series: Array<{ data: Array<{ name: string; children?: Array<{ name: string }> }> }>;
}

function build(comp: EchartsSunburstComponent, data: SunburstNode[]): SunburstOption {
  return (comp as unknown as { buildOption(d: SunburstNode[]): SunburstOption }).buildOption(data);
}

const NESTED: SunburstNode[] = [
  { name: 'Equity', children: [{ name: 'AAPL', value: 1 }, { name: 'MSFT', value: 2 }] },
];

describe('EchartsSunburstComponent', () => {
  let fixture: ComponentFixture<EchartsSunburstComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsSunburstComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsSunburstComponent);
    fixture.detectChanges();
  });

  it('when empty, it mounts without throwing', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when data is empty, the series carries no nodes', () => {
    expect(build(fixture.componentInstance, []).series[0].data.length).toBe(0);
  });

  it('when nested nodes are given, the tree (name + children) is preserved', () => {
    const data = build(fixture.componentInstance, NESTED).series[0].data;
    expect(data.length).toBe(1);
    expect(data[0].name).toBe('Equity');
    expect(data[0].children?.length).toBe(2);
    expect(data[0].children?.[0].name).toBe('AAPL');
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
