import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  EchartsRegimeTimelineComponent,
  RegimeTimelinePoint,
} from './echarts-regime-timeline';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

describe('EchartsRegimeTimelineComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsRegimeTimelineComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsRegimeTimelineComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders a container using the height input', () => {
    const fixture = TestBed.createComponent(EchartsRegimeTimelineComponent);
    fixture.componentRef.setInput('height', 260);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector('div') as HTMLDivElement;
    expect(container.style.height).toBe('260px');
  });

  it('accepts RegimeTimelinePoint data and regimeLabels without throwing', () => {
    const fixture = TestBed.createComponent(EchartsRegimeTimelineComponent);
    const data: RegimeTimelinePoint[] = [
      { date: '2026-01-01', probabilities: [0.7, 0.2, 0.1] },
      { date: '2026-01-02', probabilities: [0.6, 0.3, 0.1] },
      { date: '2026-01-03', probabilities: [0.5, 0.3, 0.2] },
    ];
    fixture.componentRef.setInput('data', data);
    fixture.componentRef.setInput('regimeLabels', ['Bull', 'Bear', 'Sideways']);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsRegimeTimelineComponent);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });
});
