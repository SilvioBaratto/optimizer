import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  EchartsRebalancingDiffComponent,
  RebalancingDiffData,
} from './echarts-rebalancing-diff';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

describe('EchartsRebalancingDiffComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsRebalancingDiffComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsRebalancingDiffComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders a container using the height input', () => {
    const fixture = TestBed.createComponent(EchartsRebalancingDiffComponent);
    fixture.componentRef.setInput('height', 260);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector('div') as HTMLDivElement;
    expect(container.style.height).toBe('260px');
  });

  it('accepts RebalancingDiffData without throwing', () => {
    const fixture = TestBed.createComponent(EchartsRebalancingDiffComponent);
    const data: RebalancingDiffData[] = [
      { asset: 'AAPL', oldWeight: 0.1, newWeight: 0.12 },
      { asset: 'MSFT', oldWeight: 0.15, newWeight: 0.1 },
      { asset: 'GOOG', oldWeight: 0.08, newWeight: 0.08 },
    ];
    fixture.componentRef.setInput('data', data);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsRebalancingDiffComponent);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });

  it('exposes getChartInstance()', () => {
    const fixture = TestBed.createComponent(EchartsRebalancingDiffComponent);
    fixture.detectChanges();

    expect(typeof fixture.componentInstance.getChartInstance).toBe('function');
  });
});
