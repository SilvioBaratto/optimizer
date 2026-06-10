// Render-coverage spec for EchartsDrawdownComponent.
// The base spec (echarts-drawdown.spec.ts) covers mount and ngOnDestroy.
// This file exercises the buildOption path:
//   - Single-path smoke: 3-point series → type='line', xAxis.data.length=3.
//   - Drawdown values are multiplied by 100 and rounded to 2dp.
//   - Tooltip formatter produces the expected string.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  installResizeObserverStub,
  getSeriesAt,
} from '../../../testing';
import {
  EchartsDrawdownComponent,
  type DrawdownPoint,
} from './echarts-drawdown';

// ---------------------------------------------------------------------------
// Cast helper
// ---------------------------------------------------------------------------

function build(
  comp: EchartsDrawdownComponent,
  points: DrawdownPoint[],
): EChartsOption {
  return (
    comp as unknown as {
      buildOption(p: DrawdownPoint[]): EChartsOption;
    }
  ).buildOption(points);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsDrawdownComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsDrawdownComponent>;
  let comp: EchartsDrawdownComponent;

  const threePoints: DrawdownPoint[] = [
    { date: '2024-01', drawdown: 0 },
    { date: '2024-02', drawdown: -0.05 },
    { date: '2024-03', drawdown: -0.12 },
  ];

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsDrawdownComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsDrawdownComponent);
    fixture.detectChanges();
    comp = fixture.componentInstance;
  });

  // ── Series type is 'line' ─────────────────────────────────────────────────

  it('series[0].type is "line"', () => {
    const opt = build(comp, threePoints);
    expect(getSeriesAt(opt, 0)['type']).toBe('line');
  });

  // ── xAxis data length matches input ──────────────────────────────────────

  it('xAxis.data.length equals the number of input points', () => {
    const opt = build(comp, threePoints);
    const xAxis = opt.xAxis as { data: unknown[] };
    expect(xAxis.data.length).toBe(3);
  });

  // ── xAxis dates preserved ────────────────────────────────────────────────

  it('xAxis.data contains the date strings from input', () => {
    const opt = build(comp, threePoints);
    const xAxis = opt.xAxis as { data: string[] };
    expect(xAxis.data).toEqual(['2024-01', '2024-02', '2024-03']);
  });

  // ── Drawdown values are converted to percentage ───────────────────────────
  // The component multiplies each drawdown by 100 and rounds to 2dp.

  it('series data contains drawdown values converted to percentage', () => {
    const opt = build(comp, threePoints);
    const series = getSeriesAt(opt, 0) as { data: number[] };
    expect(series.data[0]).toBe(0);
    expect(series.data[1]).toBeCloseTo(-5.0, 1);
    expect(series.data[2]).toBeCloseTo(-12.0, 1);
  });

  // ── yAxis max is 0 ────────────────────────────────────────────────────────

  it('yAxis.max is 0 (drawdown chart never goes above zero)', () => {
    const opt = build(comp, threePoints);
    const yAxis = opt.yAxis as { max: number };
    expect(yAxis.max).toBe(0);
  });

  // ── Tooltip formatter ─────────────────────────────────────────────────────

  it('tooltip formatter returns "date: value%" format', () => {
    const opt = build(comp, threePoints);
    const tooltip = opt.tooltip as {
      formatter: (params: Array<{ name: string; value: number }>) => string;
    };
    const result = tooltip.formatter([{ name: '2024-02', value: -5.0 }]);
    expect(result).toContain('2024-02');
    expect(result).toContain('%');
  });

  // ── areaStyle is always defined ───────────────────────────────────────────

  it('series[0].areaStyle is always defined (fill to start)', () => {
    const opt = build(comp, threePoints);
    const series = getSeriesAt(opt, 0);
    expect(series['areaStyle']).toBeDefined();
    const areaStyle = series['areaStyle'] as { origin: string };
    expect(areaStyle.origin).toBe('start');
  });
});
