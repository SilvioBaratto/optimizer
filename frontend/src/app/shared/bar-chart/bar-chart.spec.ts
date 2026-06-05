import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { BarChartComponent, type BarData } from './bar-chart';

describe('BarChartComponent', () => {
  let fixture: ComponentFixture<BarChartComponent>;

  beforeEach(async () => {
    await configureTestBed({ imports: [BarChartComponent], withHttp: false });
    fixture = TestBed.createComponent(BarChartComponent);
  });

  it('when data is empty, it mounts with no bars', () => {
    setInput(fixture, 'data', []);
    expect(fixture.nativeElement.querySelectorAll('rect').length).toBe(0);
  });

  it('when data has rows, one rect per row is rendered', () => {
    const data: BarData[] = [
      { label: 'a', value: 0.5 },
      { label: 'b', value: -0.3 },
    ];
    setInput(fixture, 'data', data);
    expect(fixture.nativeElement.querySelectorAll('rect').length).toBe(2);
  });

  it('when all values are zero, it renders without throwing', () => {
    const data: BarData[] = [
      { label: 'a', value: 0 },
      { label: 'b', value: 0 },
    ];
    expect(() => setInput(fixture, 'data', data)).not.toThrow();
  });
});
