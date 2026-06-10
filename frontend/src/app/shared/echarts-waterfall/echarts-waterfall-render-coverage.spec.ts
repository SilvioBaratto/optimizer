// Render-coverage spec for EchartsWaterfallComponent.
// The base spec (echarts-waterfall.spec.ts) covers mount and basic data.
// This file adds branches not yet exercised:
//   - Positive value: base = running (unchanged), barValue = v.
//   - Negative value: base = running + v (shifted down), barValue = -v.
//   - categories.length > 8 → xAxis.axisLabel.rotate === 45.
//   - categories.length <= 8 → xAxis.axisLabel.rotate === 0.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  installResizeObserverStub,
  getSeriesAt,
} from '../../../testing';
import { EchartsWaterfallComponent } from './echarts-waterfall';

// ---------------------------------------------------------------------------
// Cast helper
// ---------------------------------------------------------------------------

function build(
  comp: EchartsWaterfallComponent,
  categories: string[],
  values: number[],
): EChartsOption {
  return (
    comp as unknown as {
      buildOption(c: string[], v: number[]): EChartsOption;
    }
  ).buildOption(categories, values);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsWaterfallComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsWaterfallComponent>;
  let comp: EchartsWaterfallComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsWaterfallComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsWaterfallComponent);
    fixture.detectChanges();
    comp = fixture.componentInstance;
  });

  // ── Positive value branch ─────────────────────────────────────────────────
  // series[0] = transparent base bars; series[1] = coloured bar values.
  // For v=0.1 starting from base=0: base[0]=0, barValue[0]=0.1.

  it('for a positive value, base = running and barValue = v', () => {
    const opt = build(comp, ['A'], [0.1]);
    const baseSeries = getSeriesAt(opt, 0) as {
      data: Array<{ value: number }>;
    };
    const barSeries = getSeriesAt(opt, 1) as {
      data: Array<{ value: number }>;
    };
    expect(baseSeries.data[0].value).toBe(0);   // running starts at 0
    expect(barSeries.data[0].value).toBe(0.1);  // barValue = v
  });

  // ── Negative value branch ─────────────────────────────────────────────────
  // For v=-0.2 starting from base=0: base[0] = 0 + (-0.2) = -0.2, barValue = 0.2.

  it('for a negative value, base = running + v and barValue = -v', () => {
    const opt = build(comp, ['A'], [-0.2]);
    const baseSeries = getSeriesAt(opt, 0) as {
      data: Array<{ value: number }>;
    };
    const barSeries = getSeriesAt(opt, 1) as {
      data: Array<{ value: number }>;
    };
    expect(baseSeries.data[0].value).toBeCloseTo(-0.2);  // base = running + v
    expect(barSeries.data[0].value).toBeCloseTo(0.2);    // barValue = -v
  });

  // ── Mixed positive/negative accumulation ─────────────────────────────────

  it('running total accumulates across positive then negative values', () => {
    // v=[0.3, -0.1]: after first step running=0.3; second: base=0.3+(-0.1)=0.2
    const opt = build(comp, ['A', 'B'], [0.3, -0.1]);
    const baseSeries = getSeriesAt(opt, 0) as {
      data: Array<{ value: number }>;
    };
    expect(baseSeries.data[0].value).toBeCloseTo(0);    // start
    expect(baseSeries.data[1].value).toBeCloseTo(0.2);  // running(0.3) + (-0.1)
  });

  // ── axisLabel.rotate branch: <= 8 categories ─────────────────────────────

  it('when categories.length <= 8, xAxis.axisLabel.rotate is 0', () => {
    const cats = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']; // exactly 8
    const opt = build(comp, cats, [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]);
    const xAxis = opt.xAxis as { axisLabel: { rotate: number } };
    expect(xAxis.axisLabel.rotate).toBe(0);
  });

  // ── axisLabel.rotate branch: > 8 categories ──────────────────────────────

  it('when categories.length > 8, xAxis.axisLabel.rotate is 45', () => {
    const cats = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']; // 9 items
    const vals = cats.map(() => 0.1);
    const opt = build(comp, cats, vals);
    const xAxis = opt.xAxis as { axisLabel: { rotate: number } };
    expect(xAxis.axisLabel.rotate).toBe(45);
  });

  // ── Tooltip formatter ─────────────────────────────────────────────────────

  it('tooltip formatter returns signed string for positive value', () => {
    const opt = build(comp, ['Alpha'], [0.5]);
    const tooltip = opt.tooltip as {
      formatter: (params: Array<{ name: string; dataIndex: number }>) => string;
    };
    const result = tooltip.formatter([{ name: 'Alpha', dataIndex: 0 }]);
    expect(result).toContain('Alpha');
    expect(result).toContain('+');
  });

  it('tooltip formatter returns signed string for negative value', () => {
    const opt = build(comp, ['Beta'], [-0.3]);
    const tooltip = opt.tooltip as {
      formatter: (params: Array<{ name: string; dataIndex: number }>) => string;
    };
    const result = tooltip.formatter([{ name: 'Beta', dataIndex: 0 }]);
    expect(result).toContain('Beta');
    expect(result).not.toContain('+');
  });
});
