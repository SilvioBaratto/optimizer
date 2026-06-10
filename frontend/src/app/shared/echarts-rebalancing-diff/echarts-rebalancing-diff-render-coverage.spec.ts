// Render-coverage spec for EchartsRebalancingDiffComponent.
// The base spec (echarts-rebalancing-diff.spec.ts) covers mount and ngOnDestroy.
// This file exercises buildOption branches:
//   - Legend data contains 'Old' and 'New'.
//   - series[0].name === 'Old'; series[1].name === 'New'.
//   - Delta >= 0 → gainColor applied to New bar itemStyle.
//   - Delta < 0 → lossColor applied to New bar itemStyle.
//   - Old weights are converted to percentages (×100, 2dp).

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  installResizeObserverStub,
  getSeriesAt,
  getLegend,
} from '../../../testing';
import {
  EchartsRebalancingDiffComponent,
  type RebalancingDiffData,
} from './echarts-rebalancing-diff';

// ---------------------------------------------------------------------------
// Cast helper
// ---------------------------------------------------------------------------

function build(
  comp: EchartsRebalancingDiffComponent,
  rows: RebalancingDiffData[],
): EChartsOption {
  return (
    comp as unknown as {
      buildOption(r: RebalancingDiffData[]): EChartsOption;
    }
  ).buildOption(rows);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsRebalancingDiffComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsRebalancingDiffComponent>;
  let comp: EchartsRebalancingDiffComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsRebalancingDiffComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsRebalancingDiffComponent);
    fixture.detectChanges();
    comp = fixture.componentInstance;
  });

  const mixedRows: RebalancingDiffData[] = [
    { asset: 'AAPL', oldWeight: 0.1, newWeight: 0.15 },  // delta +0.05 → gain
    { asset: 'MSFT', oldWeight: 0.2, newWeight: 0.12 },  // delta -0.08 → loss
    { asset: 'GOOG', oldWeight: 0.05, newWeight: 0.05 }, // delta 0 → gain (>=0)
  ];

  // ── Legend names ──────────────────────────────────────────────────────────

  it('legend.data contains "Old" and "New"', () => {
    const opt = build(comp, mixedRows);
    const legend = getLegend(opt) as { data: string[] };
    expect(legend.data).toContain('Old');
    expect(legend.data).toContain('New');
  });

  // ── Series names ──────────────────────────────────────────────────────────

  it('series[0].name is "Old"', () => {
    const opt = build(comp, mixedRows);
    expect(getSeriesAt(opt, 0)['name']).toBe('Old');
  });

  it('series[1].name is "New"', () => {
    const opt = build(comp, mixedRows);
    expect(getSeriesAt(opt, 1)['name']).toBe('New');
  });

  // ── Both series type is 'bar' ─────────────────────────────────────────────

  it('both series use type "bar"', () => {
    const opt = build(comp, mixedRows);
    expect(getSeriesAt(opt, 0)['type']).toBe('bar');
    expect(getSeriesAt(opt, 1)['type']).toBe('bar');
  });

  // ── Old weights converted to percentage ───────────────────────────────────

  it('series[0] data contains old weights as percentages', () => {
    const opt = build(comp, mixedRows);
    const series = getSeriesAt(opt, 0) as { data: number[] };
    expect(series.data[0]).toBeCloseTo(10.0, 1);  // 0.1 * 100
    expect(series.data[1]).toBeCloseTo(20.0, 1);  // 0.2 * 100
  });

  // ── Delta >= 0 → gainColor on New bar ────────────────────────────────────
  // JSDOM resolves CSS vars to '' so gainColor === '' in test env.

  it('when delta >= 0, New bar itemStyle.color uses gainColor', () => {
    const opt = build(comp, [
      { asset: 'AAPL', oldWeight: 0.1, newWeight: 0.15 }, // +5%
    ]);
    const newSeries = getSeriesAt(opt, 1) as {
      data: Array<{ value: number; itemStyle: { color: string } }>;
    };
    const style = getComputedStyle(document.documentElement);
    const gainColor = style.getPropertyValue('--color-gain').trim();
    expect(newSeries.data[0].itemStyle.color).toBe(gainColor);
  });

  it('when delta === 0, New bar itemStyle.color uses gainColor (>= 0 branch)', () => {
    const opt = build(comp, [
      { asset: 'GOOG', oldWeight: 0.05, newWeight: 0.05 }, // delta=0
    ]);
    const newSeries = getSeriesAt(opt, 1) as {
      data: Array<{ value: number; itemStyle: { color: string } }>;
    };
    const style = getComputedStyle(document.documentElement);
    const gainColor = style.getPropertyValue('--color-gain').trim();
    expect(newSeries.data[0].itemStyle.color).toBe(gainColor);
  });

  // ── Delta < 0 → lossColor on New bar ─────────────────────────────────────

  it('when delta < 0, New bar itemStyle.color uses lossColor', () => {
    const opt = build(comp, [
      { asset: 'MSFT', oldWeight: 0.2, newWeight: 0.12 }, // -8%
    ]);
    const newSeries = getSeriesAt(opt, 1) as {
      data: Array<{ value: number; itemStyle: { color: string } }>;
    };
    const style = getComputedStyle(document.documentElement);
    const lossColor = style.getPropertyValue('--color-loss').trim();
    expect(newSeries.data[0].itemStyle.color).toBe(lossColor);
  });

  // ── Label formatter ───────────────────────────────────────────────────────

  it('New series label formatter prefixes positive delta with "+"', () => {
    const opt = build(comp, [
      { asset: 'AAPL', oldWeight: 0.1, newWeight: 0.15 }, // +5%
    ]);
    const newSeries = getSeriesAt(opt, 1) as {
      label: { formatter: (p: { dataIndex: number }) => string };
    };
    const result = newSeries.label.formatter({ dataIndex: 0 });
    expect(result).toContain('+');
    expect(result).toContain('%');
  });

  it('New series label formatter does not prefix negative delta with "+"', () => {
    const opt = build(comp, [
      { asset: 'MSFT', oldWeight: 0.2, newWeight: 0.12 }, // -8%
    ]);
    const newSeries = getSeriesAt(opt, 1) as {
      label: { formatter: (p: { dataIndex: number }) => string };
    };
    const result = newSeries.label.formatter({ dataIndex: 0 });
    expect(result).not.toContain('+');
    expect(result).toContain('%');
  });
});
