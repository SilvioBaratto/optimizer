import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  EchartsFactorExposureComponent,
  FactorExposureItem,
} from './echarts-factor-exposure';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

describe('EchartsFactorExposureComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsFactorExposureComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsFactorExposureComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders a container using the height input', () => {
    const fixture = TestBed.createComponent(EchartsFactorExposureComponent);
    fixture.componentRef.setInput('height', 240);
    fixture.detectChanges();

    const container = fixture.nativeElement.querySelector('div') as HTMLDivElement;
    expect(container.style.height).toBe('240px');
  });

  it('accepts FactorExposureItem data without throwing', () => {
    const fixture = TestBed.createComponent(EchartsFactorExposureComponent);
    const data: FactorExposureItem[] = [
      { factor: 'Momentum', exposure: 0.32, group: 'Style' },
      { factor: 'Value', exposure: -0.18, group: 'Style' },
      { factor: 'Quality', exposure: 0.12 },
    ];
    fixture.componentRef.setInput('data', data);
    expect(() => fixture.detectChanges()).not.toThrow();
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsFactorExposureComponent);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });
});
