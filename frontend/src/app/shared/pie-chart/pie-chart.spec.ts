import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { PieChartComponent, type PieSegment } from './pie-chart';

describe('PieChartComponent', () => {
  let fixture: ComponentFixture<PieChartComponent>;
  let comp: PieChartComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [PieChartComponent], withHttp: false });
    fixture = TestBed.createComponent(PieChartComponent);
    comp = fixture.componentInstance;
  });

  it('when segments are empty, no arcs are produced', () => {
    setInput(fixture, 'segments', []);
    expect(comp.arcs()).toEqual([]);
    expect(fixture.nativeElement.querySelectorAll('path').length).toBe(0);
  });

  it('when all values are zero, the total guard yields no arcs', () => {
    const segs: PieSegment[] = [
      { label: 'a', value: 0, color: '#000' },
      { label: 'b', value: 0, color: '#111' },
    ];
    setInput(fixture, 'segments', segs);
    expect(comp.arcs()).toEqual([]);
  });

  it('when two equal segments are given, each is 50% with a path', () => {
    const segs: PieSegment[] = [
      { label: 'a', value: 1, color: '#000' },
      { label: 'b', value: 1, color: '#111' },
    ];
    setInput(fixture, 'segments', segs);
    const arcs = comp.arcs();
    expect(arcs.length).toBe(2);
    expect(arcs[0].pct).toBe(50);
    expect(arcs[1].pct).toBe(50);
    expect(fixture.nativeElement.querySelectorAll('path').length).toBe(2);
  });
});
