import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { LineChartComponent, type LineChartData } from './line-chart';

// `createChart` is a frozen ESM export from lightweight-charts and cannot be
// spied. The lib runs fine in headless Chrome, so we drive the real
// chart and assert it builds/tears down DOM in the container.
describe('LineChartComponent', () => {
  let fixture: ComponentFixture<LineChartComponent>;

  function container(): HTMLElement {
    return fixture.nativeElement.querySelector('div') as HTMLElement;
  }

  function init(): void {
    fixture.detectChanges(); // resolve the required viewChild(container)
    (fixture.componentInstance as unknown as { initChart(): void }).initChart();
  }

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [LineChartComponent], withHttp: false });
    fixture = TestBed.createComponent(LineChartComponent);
  });

  it('when initialised, the chart builds DOM inside the container', () => {
    init();
    expect(container().children.length).toBeGreaterThan(0);
  });

  it('when data is provided, it initialises without throwing', () => {
    const data: LineChartData[] = [
      { time: '2026-01-01', value: 1 },
      { time: '2026-01-02', value: 2 },
    ];
    fixture.componentRef.setInput('data', data);
    expect(() => init()).not.toThrow();
  });

  it('when destroyed after init, the chart is removed without throwing', () => {
    init();
    expect(() => fixture.destroy()).not.toThrow();
  });

  it('when destroyed before init, ngOnDestroy is null-safe', () => {
    expect(() => fixture.componentInstance.ngOnDestroy()).not.toThrow();
  });
});
