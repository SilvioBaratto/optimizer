import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { EchartsDonutComponent, PieSegment } from './echarts-donut';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

const NARROW_WIDTH = 300;
const WIDE_WIDTH = 900;

function manySegments(n: number): PieSegment[] {
  return Array.from({ length: n }, (_, i) => ({
    label: `seg-${i}`,
    value: i + 1,
    color: '#000',
  }));
}

describe('EchartsDonutComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsDonutComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  function createComponent(): EchartsDonutComponent {
    const fixture = TestBed.createComponent(EchartsDonutComponent);
    fixture.componentRef.setInput('segments', []);
    fixture.detectChanges();
    return fixture.componentInstance;
  }

  function seriesLabelShow(option: unknown): unknown {
    const series = (option as { series: Array<{ label?: { show?: unknown } }> }).series;
    return series[0].label!.show;
  }

  function legendType(option: unknown): unknown {
    const legend = (option as { legend: { type?: unknown } }).legend;
    return legend.type;
  }

  function tooltipFormatter(option: unknown): unknown {
    const tooltip = (option as { tooltip: { formatter?: unknown } }).tooltip;
    return tooltip.formatter;
  }

  it('creates successfully', () => {
    const component = createComponent();
    expect(component).toBeTruthy();
  });

  it('defaults labelThreshold to 8', () => {
    const component = createComponent();
    expect(component.labelThreshold()).toBe(8);
  });

  it('narrow + segments ≤ threshold → series[0].label.show === true', () => {
    const component = createComponent();
    const option = component.buildOptionForTest(manySegments(5), NARROW_WIDTH);
    expect(seriesLabelShow(option)).toBe(true);
  });

  it('narrow + segments > threshold (20) → series[0].label.show === false', () => {
    const component = createComponent();
    const option = component.buildOptionForTest(manySegments(20), NARROW_WIDTH);
    expect(seriesLabelShow(option)).toBe(false);
  });

  it('wide + segments ≤ threshold → series[0].label.show === false (desktop behaviour preserved)', () => {
    const component = createComponent();
    const option = component.buildOptionForTest(manySegments(5), WIDE_WIDTH);
    expect(seriesLabelShow(option)).toBe(false);
  });

  it('segments > threshold on a narrow container → legend.type === "scroll"', () => {
    const component = createComponent();
    const option = component.buildOptionForTest(manySegments(20), NARROW_WIDTH);
    expect(legendType(option)).toBe('scroll');
  });

  it('segments > threshold on a wide container → legend.type === "scroll"', () => {
    const component = createComponent();
    const option = component.buildOptionForTest(manySegments(20), WIDE_WIDTH);
    expect(legendType(option)).toBe('scroll');
  });

  it('segments ≤ threshold on a narrow container → legend.type is NOT "scroll"', () => {
    const component = createComponent();
    const option = component.buildOptionForTest(manySegments(8), NARROW_WIDTH);
    expect(legendType(option)).not.toBe('scroll');
  });

  it('segments ≤ threshold on a wide container → legend.type is NOT "scroll"', () => {
    const component = createComponent();
    const option = component.buildOptionForTest(manySegments(8), WIDE_WIDTH);
    expect(legendType(option)).not.toBe('scroll');
  });

  it('tooltip formatter is {b}: {d}% regardless of segment count or width', () => {
    const component = createComponent();
    const cases = [
      { segs: manySegments(3), width: NARROW_WIDTH },
      { segs: manySegments(3), width: WIDE_WIDTH },
      { segs: manySegments(20), width: NARROW_WIDTH },
      { segs: manySegments(20), width: WIDE_WIDTH },
    ];
    for (const c of cases) {
      const option = component.buildOptionForTest(c.segs, c.width);
      expect(tooltipFormatter(option)).toBe('{b}: {d}%');
    }
  });

  it('respects a caller-specified labelThreshold', () => {
    const fixture = TestBed.createComponent(EchartsDonutComponent);
    fixture.componentRef.setInput('segments', []);
    fixture.componentRef.setInput('labelThreshold', 3);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest(
      manySegments(5),
      NARROW_WIDTH,
    );
    expect(seriesLabelShow(option)).toBe(false);
    expect(legendType(option)).toBe('scroll');
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsDonutComponent);
    fixture.componentRef.setInput('segments', []);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });
});
