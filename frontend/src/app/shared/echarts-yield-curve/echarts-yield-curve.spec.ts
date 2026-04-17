import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  EchartsYieldCurveComponent,
  YieldCurveData,
} from './echarts-yield-curve';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

describe('EchartsYieldCurveComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsYieldCurveComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsYieldCurveComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders a container using the height input', () => {
    const fixture = TestBed.createComponent(EchartsYieldCurveComponent);
    fixture.componentRef.setInput('height', 300);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector('div') as HTMLDivElement;
    expect(container.style.height).toBe('300px');
  });

  it('accepts multiple YieldCurveData series without throwing', () => {
    const fixture = TestBed.createComponent(EchartsYieldCurveComponent);
    const data: YieldCurveData[] = [
      {
        label: 'Today',
        maturities: ['3M', '1Y', '5Y', '10Y', '30Y'],
        yields: [4.2, 4.0, 3.8, 3.9, 4.0],
      },
      {
        label: '1Y ago',
        maturities: ['3M', '1Y', '5Y', '10Y', '30Y'],
        yields: [3.0, 3.2, 3.5, 3.6, 3.7],
      },
    ];
    fixture.componentRef.setInput('data', data);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsYieldCurveComponent);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });

  it('exposes getChartInstance()', () => {
    const fixture = TestBed.createComponent(EchartsYieldCurveComponent);
    fixture.detectChanges();

    expect(typeof fixture.componentInstance.getChartInstance).toBe('function');
  });
});
