import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsScatterComponent, type ScatterPoint } from './echarts-scatter';

interface ScatterOption {
  series: unknown[];
}

function build(
  comp: EchartsScatterComponent,
  pts: ScatterPoint[],
  optimal: ScatterPoint | null,
  highlighted: ScatterPoint[],
): ScatterOption {
  return (
    comp as unknown as {
      buildOption(p: ScatterPoint[], o: ScatterPoint | null, h: ScatterPoint[]): ScatterOption;
    }
  ).buildOption(pts, optimal, highlighted);
}

const FRONTIER: ScatterPoint[] = [
  { x: 0.1, y: 0.05 },
  { x: 0.2, y: 0.08 },
];

describe('EchartsScatterComponent', () => {
  let fixture: ComponentFixture<EchartsScatterComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsScatterComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsScatterComponent);
    fixture.detectChanges();
  });

  it('when empty, it mounts without throwing', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('when only the frontier is given, there is a single line series', () => {
    expect(build(fixture.componentInstance, FRONTIER, null, []).series.length).toBe(1);
  });

  it('when an optimal point is added, a scatter series is appended', () => {
    expect(build(fixture.componentInstance, FRONTIER, { x: 0.15, y: 0.07 }, []).series.length).toBe(2);
  });

  it('when optimal and highlighted points are added, the series count grows', () => {
    const opt = build(fixture.componentInstance, FRONTIER, { x: 0.15, y: 0.07 }, [
      { x: 0.18, y: 0.06 },
      { x: 0.12, y: 0.04 },
    ]);
    expect(opt.series.length).toBe(4); // 1 frontier + 1 optimal + 2 highlighted
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
