import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  EchartsCandlestickComponent,
  CandlestickData,
} from './echarts-candlestick';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

describe('EchartsCandlestickComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsCandlestickComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsCandlestickComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders a container using the height input', () => {
    const fixture = TestBed.createComponent(EchartsCandlestickComponent);
    fixture.componentRef.setInput('height', 360);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector('div') as HTMLDivElement;
    expect(container.style.height).toBe('360px');
  });

  it('accepts OHLC data without volume and does not throw', () => {
    const fixture = TestBed.createComponent(EchartsCandlestickComponent);
    const data: CandlestickData[] = [
      { date: '2026-01-01', open: 100, close: 105, high: 106, low: 99 },
      { date: '2026-01-02', open: 105, close: 103, high: 107, low: 102 },
    ];
    fixture.componentRef.setInput('data', data);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('accepts OHLC data with volume and does not throw', () => {
    const fixture = TestBed.createComponent(EchartsCandlestickComponent);
    const data: CandlestickData[] = [
      { date: '2026-01-01', open: 100, close: 105, high: 106, low: 99, volume: 1_000_000 },
      { date: '2026-01-02', open: 105, close: 103, high: 107, low: 102, volume: 1_200_000 },
    ];
    fixture.componentRef.setInput('data', data);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsCandlestickComponent);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });

  it('exposes getChartInstance()', () => {
    const fixture = TestBed.createComponent(EchartsCandlestickComponent);
    fixture.detectChanges();

    expect(typeof fixture.componentInstance.getChartInstance).toBe('function');
  });
});
