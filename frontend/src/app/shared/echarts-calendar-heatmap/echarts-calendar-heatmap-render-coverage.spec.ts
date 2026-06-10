// Render-coverage spec for EchartsCalendarHeatmapComponent.
// The base spec (echarts-calendar-heatmap.spec.ts) covers mount and ngOnDestroy.
// This file exercises buildOption branches:
//   - absMax===0 guard: all-zero data → visualMap.max===0.01 and min===-0.01.
//   - Normal data: visualMap.max equals the actual absMax value.
//   - Series type is 'heatmap'.
//   - xAxis.data and yAxis.data match the months/years inputs.
//   - Tooltip formatter returns '' for zero values.
//   - Tooltip formatter returns signed percentage for non-zero values.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  installResizeObserverStub,
  getSeriesAt,
} from '../../../testing';
import { EchartsCalendarHeatmapComponent } from './echarts-calendar-heatmap';

// ---------------------------------------------------------------------------
// Cast helper
// ---------------------------------------------------------------------------

function build(
  comp: EchartsCalendarHeatmapComponent,
  years: string[],
  months: string[],
  data: number[][],
): EChartsOption {
  return (
    comp as unknown as {
      buildOption(y: string[], m: string[], d: number[][]): EChartsOption;
    }
  ).buildOption(years, months, data);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsCalendarHeatmapComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsCalendarHeatmapComponent>;
  let comp: EchartsCalendarHeatmapComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsCalendarHeatmapComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsCalendarHeatmapComponent);
    fixture.detectChanges();
    comp = fixture.componentInstance;
  });

  // ── absMax === 0 guard ────────────────────────────────────────────────────

  it('when all data values are zero, visualMap.max is 0.01 and min is -0.01', () => {
    const opt = build(comp, ['2024'], ['Jan', 'Feb'], [[0, 0]]);
    const vm = opt.visualMap as { min: number; max: number };
    expect(vm.max).toBeCloseTo(0.01);
    expect(vm.min).toBeCloseTo(-0.01);
  });

  // ── Normal data: visualMap bounds reflect actual absMax ───────────────────

  it('when data has non-zero values, visualMap.max equals the actual absMax', () => {
    // absMax = 0.05 (from value 0.05)
    const opt = build(comp, ['2024'], ['Jan', 'Feb'], [[0.03, -0.05]]);
    const vm = opt.visualMap as { min: number; max: number };
    expect(vm.max).toBeCloseTo(0.05);
    expect(vm.min).toBeCloseTo(-0.05);
  });

  it('when data has a large positive value, visualMap bounds are symmetric', () => {
    const opt = build(comp, ['2023', '2024'], ['Jan'], [[0.08], [0.02]]);
    const vm = opt.visualMap as { min: number; max: number };
    expect(vm.max).toBeCloseTo(0.08);
    expect(vm.min).toBeCloseTo(-0.08);
  });

  // ── Series type is 'heatmap' ──────────────────────────────────────────────

  it('series[0].type is "heatmap"', () => {
    const opt = build(comp, ['2024'], ['Jan'], [[0.01]]);
    expect(getSeriesAt(opt, 0)['type']).toBe('heatmap');
  });

  // ── xAxis and yAxis reflect months/years ─────────────────────────────────

  it('xAxis.data matches the months input', () => {
    const months = ['Jan', 'Feb', 'Mar'];
    const opt = build(comp, ['2024'], months, [[0.01, 0.02, 0.03]]);
    const xAxis = opt.xAxis as { data: string[] };
    expect(xAxis.data).toEqual(months);
  });

  it('yAxis.data matches the years input', () => {
    const years = ['2022', '2023', '2024'];
    const opt = build(
      comp,
      years,
      ['Jan'],
      [[0.01], [0.02], [0.03]],
    );
    const yAxis = opt.yAxis as { data: string[] };
    expect(yAxis.data).toEqual(years);
  });

  // ── Tooltip formatter: zero value returns '' ──────────────────────────────

  it('tooltip formatter returns empty string when value is 0', () => {
    const opt = build(comp, ['2024'], ['Jan', 'Feb'], [[0, 0]]);
    type TooltipFn = { formatter: (params: { value: [number, number, number] }) => string };
    const tooltip = (opt.tooltip as unknown) as TooltipFn;
    const result = tooltip.formatter({ value: [0, 0, 0] });
    expect(result).toBe('');
  });

  // ── Tooltip formatter: non-zero value returns signed percentage ───────────

  it('tooltip formatter returns signed percentage for positive value', () => {
    const opt = build(comp, ['2024'], ['Jan', 'Feb'], [[0.05, 0]]);
    type TooltipFn = { formatter: (params: { value: [number, number, number] }) => string };
    const tooltip = (opt.tooltip as unknown) as TooltipFn;
    // tuple: [colIndex, rowIndex, value] = [0, 0, 0.05]
    const result = tooltip.formatter({ value: [0, 0, 0.05] });
    expect(result).toContain('+');
    expect(result).toContain('%');
    expect(result).toContain('Jan');
    expect(result).toContain('2024');
  });

  it('tooltip formatter returns signed percentage for negative value', () => {
    const opt = build(comp, ['2024'], ['Jan', 'Feb'], [[0, -0.03]]);
    type TooltipFn = { formatter: (params: { value: [number, number, number] }) => string };
    const tooltip = (opt.tooltip as unknown) as TooltipFn;
    // tuple: [colIndex=1, rowIndex=0, value=-0.03]
    const result = tooltip.formatter({ value: [1, 0, -0.03] });
    expect(result).not.toContain('+');
    expect(result).toContain('%');
    expect(result).toContain('Feb');
    expect(result).toContain('2024');
  });

  // ── Missing data row defaults to 0 ───────────────────────────────────────
  // The component uses `data[row]?.[col] ?? 0` for missing rows.

  it('when a data row is missing, the value defaults to 0 and absMax guard fires', () => {
    // 2 years but only 1 data row — row index 1 is missing
    const opt = build(comp, ['2023', '2024'], ['Jan'], [[0]]);
    const vm = opt.visualMap as { min: number; max: number };
    // Both values are 0 → absMax guard sets to 0.01
    expect(vm.max).toBeCloseTo(0.01);
  });
});
