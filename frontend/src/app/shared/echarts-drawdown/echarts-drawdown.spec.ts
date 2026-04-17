import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  EchartsDrawdownComponent,
  DrawdownPoint,
} from './echarts-drawdown';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

describe('EchartsDrawdownComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsDrawdownComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsDrawdownComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders a container using the height input', () => {
    const fixture = TestBed.createComponent(EchartsDrawdownComponent);
    fixture.componentRef.setInput('height', 300);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector(
      'div',
    ) as HTMLDivElement;
    expect(container).not.toBeNull();
    expect(container.style.height).toBe('300px');
  });

  it('accepts DrawdownPoint data without throwing', () => {
    const fixture = TestBed.createComponent(EchartsDrawdownComponent);
    const data: DrawdownPoint[] = [
      { date: '2026-01-01', drawdown: 0 },
      { date: '2026-01-02', drawdown: -0.05 },
      { date: '2026-01-03', drawdown: -0.15 },
    ];
    fixture.componentRef.setInput('data', data);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsDrawdownComponent);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });
});
