import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { EchartsEfficientFrontierComponent } from './echarts-efficient-frontier';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';
import type { EfficientFrontierPoint } from '../../core/models/optimization.model';

function makeFrontier(): EfficientFrontierPoint[] {
  return [
    { risk: 0.05, return: 0.03, sharpe: 0.6 },
    { risk: 0.08, return: 0.06, sharpe: 0.75, label: 'mid' },
    { risk: 0.12, return: 0.09, sharpe: 0.75 },
  ];
}

describe('EchartsEfficientFrontierComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsEfficientFrontierComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', []);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders the empty state when frontier is empty', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', []);
    fixture.detectChanges();

    const emptyNode = fixture.nativeElement.querySelector(
      '[data-testid="efficient-frontier-empty"]',
    );
    expect(emptyNode).not.toBeNull();
  });

  it('does not render the chart container when frontier is empty', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', []);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector(
      '[data-testid="efficient-frontier-chart"]',
    );
    expect(container).toBeNull();
  });

  it('renders the chart container when frontier has points', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector(
      '[data-testid="efficient-frontier-chart"]',
    );
    expect(container).not.toBeNull();
  });

  it('uses the height input on the chart container', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.componentRef.setInput('height', 420);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector(
      '[data-testid="efficient-frontier-chart"]',
    ) as HTMLDivElement;
    expect(container.style.height).toBe('420px');
  });

  it('builds a single line series for the frontier when optimal/markers absent', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const series = option['series'] as Array<{ type: string; name?: string }>;
    expect(series.length).toBe(1);
    expect(series[0].type).toBe('line');
  });

  it('adds scatter series for optimal and asset markers when provided', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    const optimal: EfficientFrontierPoint = {
      risk: 0.08,
      return: 0.06,
      sharpe: 0.75,
      label: 'Max Sharpe',
    };
    const markers: EfficientFrontierPoint[] = [
      { risk: 0.05, return: 0.03, sharpe: 0.6, label: 'Min Variance' },
    ];
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.componentRef.setInput('optimal', optimal);
    fixture.componentRef.setInput('assetMarkers', markers);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const series = option['series'] as Array<{ type: string; name?: string }>;
    const types = series.map((s) => s.type);
    expect(types.filter((t) => t === 'line').length).toBe(1);
    expect(types.filter((t) => t === 'scatter').length).toBe(2);
  });

  it('includes a CAL line series when riskFreeRate is provided', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.componentRef.setInput('riskFreeRate', 0.02);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const series = option['series'] as Array<{ type: string; name?: string }>;
    const calSeries = series.filter((s) => s.name === 'Capital Allocation Line');
    expect(calSeries.length).toBe(1);
    expect(calSeries[0].type).toBe('line');
  });

  it('omits the CAL line when riskFreeRate is null', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.componentRef.setInput('riskFreeRate', null);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const series = option['series'] as Array<{ name?: string }>;
    const calSeries = series.filter((s) => s.name === 'Capital Allocation Line');
    expect(calSeries.length).toBe(0);
  });

  it('emits pointClick with the matching EfficientFrontierPoint on frontier click', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    const frontier = makeFrontier();
    fixture.componentRef.setInput('frontier', frontier);

    const events: EfficientFrontierPoint[] = [];
    fixture.componentInstance.pointClick.subscribe((p) => events.push(p));

    fixture.componentInstance.handleFrontierClickForTest({
      seriesName: 'Efficient Frontier',
      dataIndex: 1,
    });

    expect(events.length).toBe(1);
    expect(events[0]).toEqual(frontier[1]);
  });

  it('does not emit pointClick for a non-frontier series click', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', makeFrontier());

    const events: EfficientFrontierPoint[] = [];
    fixture.componentInstance.pointClick.subscribe((p) => events.push(p));

    fixture.componentInstance.handleFrontierClickForTest({
      seriesName: 'Max Sharpe',
      dataIndex: 0,
    });

    expect(events.length).toBe(0);
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
    fixture.componentRef.setInput('frontier', makeFrontier());
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });
});
