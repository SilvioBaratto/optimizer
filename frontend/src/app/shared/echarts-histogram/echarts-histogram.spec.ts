import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import { EchartsHistogramComponent } from './echarts-histogram';

interface HistogramOption {
  series?: unknown[];
}

function build(comp: EchartsHistogramComponent, values: number[]): HistogramOption {
  return (comp as unknown as { buildOption(v: number[]): HistogramOption }).buildOption(values);
}

const VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

describe('EchartsHistogramComponent', () => {
  let fixture: ComponentFixture<EchartsHistogramComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsHistogramComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsHistogramComponent);
    fixture.detectChanges();
  });

  it('when values are empty, buildOption returns an empty option', () => {
    expect(build(fixture.componentInstance, []).series).toBeUndefined();
  });

  it('when values are populated, a single bar series is produced', () => {
    expect(build(fixture.componentInstance, VALUES).series!.length).toBe(1);
  });

  it('when overlayNormal is on, a second (line) series is appended', () => {
    fixture.componentRef.setInput('overlayNormal', true);
    fixture.detectChanges();
    expect(build(fixture.componentInstance, VALUES).series!.length).toBe(2);
  });

  it('when all values are equal, the zero-range guard yields a single bar', () => {
    expect(build(fixture.componentInstance, [5, 5, 5]).series!.length).toBe(1);
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });

  it('when injected, it registers as CHART_EXPORTABLE', () => {
    expect(fixture.debugElement.injector.get(CHART_EXPORTABLE)).toBe(fixture.componentInstance);
  });
});
