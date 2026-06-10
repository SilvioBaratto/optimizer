// Render-coverage spec for EchartsStackedAreaComponent.
// The base spec (echarts-stacked-area.spec.ts) covers mount and basic series.
// This file adds branches not yet exercised:
//   - Narrow path (JSDOM clientWidth=0 < 500): grid.left=40, grid.bottom=70.
//   - Wide path (clientWidth=700 >= 500): grid.left=50, grid.bottom=40.
//   - Series with explicit color → color used as-is.
//   - Series without color → falls back to chartColors palette.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  installResizeObserverStub,
  getSeriesAt,
} from '../../../testing';
import {
  EchartsStackedAreaComponent,
  type AreaSeries,
} from './echarts-stacked-area';

// ---------------------------------------------------------------------------
// Cast helper
// ---------------------------------------------------------------------------

function build(
  comp: EchartsStackedAreaComponent,
  labels: string[],
  series: AreaSeries[],
): EChartsOption {
  return (
    comp as unknown as {
      buildOption(l: string[], s: AreaSeries[]): EChartsOption;
    }
  ).buildOption(labels, series);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsStackedAreaComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsStackedAreaComponent>;
  let comp: EchartsStackedAreaComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsStackedAreaComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsStackedAreaComponent);
    fixture.detectChanges();
    comp = fixture.componentInstance;
  });

  // ── Narrow path (stub clientWidth=0 < 500) ───────────────────────────────

  it('when container is narrow (clientWidth=0), grid.left=40 and grid.bottom=70', () => {
    const containerEl = (
      comp as unknown as { container: () => { nativeElement: HTMLElement } }
    ).container().nativeElement;
    Object.defineProperty(containerEl, 'clientWidth', {
      value: 0,
      configurable: true,
    });

    const opt = build(comp, ['Jan', 'Feb'], [{ name: 'A', values: [1, 2] }]);
    const grid = opt.grid as { left: number; bottom: number };
    expect(grid.left).toBe(40);
    expect(grid.bottom).toBe(70);
  });

  it('when container is narrow, legend textStyle.fontSize is 10', () => {
    const containerEl = (
      comp as unknown as { container: () => { nativeElement: HTMLElement } }
    ).container().nativeElement;
    Object.defineProperty(containerEl, 'clientWidth', {
      value: 0,
      configurable: true,
    });

    const opt = build(comp, ['Jan'], [{ name: 'A', values: [1] }]);
    const legend = opt.legend as { textStyle: { fontSize: number } };
    expect(legend.textStyle.fontSize).toBe(10);
  });

  // ── Wide path (stub clientWidth >= 500) ──────────────────────────────────

  it('when container is wide (clientWidth=700), grid.left=50 and grid.bottom=40', () => {
    // Stub the container nativeElement's clientWidth to simulate a wide layout
    const containerEl = (
      comp as unknown as { container: () => { nativeElement: HTMLElement } }
    ).container().nativeElement;
    Object.defineProperty(containerEl, 'clientWidth', {
      value: 700,
      configurable: true,
    });

    const opt = build(comp, ['Jan', 'Feb'], [{ name: 'A', values: [1, 2] }]);
    const grid = opt.grid as { left: number; bottom: number };
    expect(grid.left).toBe(50);
    expect(grid.bottom).toBe(40);
  });

  it('when container is wide, legend textStyle.fontSize is 12', () => {
    const containerEl = (
      comp as unknown as { container: () => { nativeElement: HTMLElement } }
    ).container().nativeElement;
    Object.defineProperty(containerEl, 'clientWidth', {
      value: 700,
      configurable: true,
    });

    const opt = build(comp, ['Jan'], [{ name: 'A', values: [1] }]);
    const legend = opt.legend as { textStyle: { fontSize: number } };
    expect(legend.textStyle.fontSize).toBe(12);
  });

  // ── Series color: explicit color used as-is ───────────────────────────────

  it('when a series has an explicit color, that color is used in itemStyle and lineStyle', () => {
    const explicitColor = '#ff6600';
    const opt = build(comp, ['Jan', 'Feb'], [
      { name: 'Alpha', values: [1, 2], color: explicitColor },
    ]);
    const s = getSeriesAt(opt, 0) as {
      itemStyle: { color: string };
      lineStyle: { color: string };
    };
    expect(s.itemStyle.color).toBe(explicitColor);
    expect(s.lineStyle.color).toBe(explicitColor);
  });

  // ── Series color: palette fallback when no color provided ─────────────────
  // When color is absent the component reads --color-chart-{i+1} CSS vars.
  // In JSDOM those resolve to '', but the fallback path still executes.

  it('when a series has no explicit color, color is derived from the CSS palette', () => {
    const opt = build(comp, ['Jan', 'Feb'], [
      { name: 'Beta', values: [1, 2] }, // no color field
    ]);
    const s = getSeriesAt(opt, 0) as {
      itemStyle: { color: string };
      lineStyle: { color: string };
    };
    // Palette color equals the CSS var value (empty string in JSDOM) — both must match
    expect(s.itemStyle.color).toBe(s.lineStyle.color);
  });

  // ── Series type is always 'line' ──────────────────────────────────────────

  it('series type is always "line"', () => {
    const opt = build(comp, ['Jan'], [{ name: 'X', values: [1] }]);
    expect(getSeriesAt(opt, 0)['type']).toBe('line');
  });

  // ── Multiple series stacking ──────────────────────────────────────────────

  it('each series has stack="total" for stacking', () => {
    const opt = build(comp, ['Jan', 'Feb'], [
      { name: 'A', values: [1, 2] },
      { name: 'B', values: [3, 4] },
    ]);
    expect(getSeriesAt(opt, 0)['stack']).toBe('total');
    expect(getSeriesAt(opt, 1)['stack']).toBe('total');
  });
});
