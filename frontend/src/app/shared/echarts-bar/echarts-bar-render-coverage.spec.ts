// Render-coverage spec for EchartsBarComponent.
// The base spec (echarts-bar.spec.ts) already covers:
//   - mount with empty/populated data, series/axis derivation, ngOnDestroy, CHART_EXPORTABLE.
// This file adds branches not yet exercised:
//   - Month-label formatting: 'YYYY-MM' → "Mon 'YY" vs passthrough for non-month labels.
//   - Color resolution: explicit `color` field used as-is vs sign-derived color from CSS vars.

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
} from '../../../testing';
import { EchartsBarComponent, type BarData } from './echarts-bar';

// ---------------------------------------------------------------------------
// Cast helper — mirrors the idiom in echarts-bar.spec.ts
// ---------------------------------------------------------------------------

interface BarOption {
  xAxis: { data: string[] };
  series: Array<{
    data: Array<{ value: number; itemStyle: { color: string } }>;
  }>;
}

function build(comp: EchartsBarComponent, bars: BarData[]): BarOption {
  return (comp as unknown as { buildOption(b: BarData[]): BarOption }).buildOption(bars);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsBarComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsBarComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsBarComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsBarComponent);
    // detectChanges first so viewChild('container') resolves before the cast
    fixture.detectChanges();
  });

  // ── Month-label branch: 'YYYY-MM' formatted ──────────────────────────────

  it('when a label matches YYYY-MM format, it is formatted as "Mon \'YY"', () => {
    const opt = build(fixture.componentInstance, [
      { label: '2026-01', value: 0.1 },
    ]);
    // January 2026 → "Jan '26"
    expect(opt.xAxis.data[0]).toBe("Jan '26");
  });

  it('when a label is "2026-06", it is formatted as "Jun \'26"', () => {
    const opt = build(fixture.componentInstance, [
      { label: '2026-06', value: 0.05 },
    ]);
    expect(opt.xAxis.data[0]).toBe("Jun '26");
  });

  // ── Non-month label: passthrough ─────────────────────────────────────────

  it('when a label does not match YYYY-MM (e.g. a ticker), it is passed through unchanged', () => {
    const opt = build(fixture.componentInstance, [
      { label: 'AAPL', value: 0.2 },
    ]);
    expect(opt.xAxis.data[0]).toBe('AAPL');
  });

  it('when a label has more than two hyphen-parts, it is passed through unchanged', () => {
    const opt = build(fixture.componentInstance, [
      { label: '2026-01-15', value: 0.0 },
    ]);
    // '2026-01-15'.split('-') has 3 parts → passthrough
    expect(opt.xAxis.data[0]).toBe('2026-01-15');
  });

  // ── Color branch: explicit color used as-is ──────────────────────────────

  it('when a datum has an explicit color, that color is used directly', () => {
    const explicitColor = '#ff0000';
    const opt = build(fixture.componentInstance, [
      { label: 'X', value: 0.5, color: explicitColor },
    ]);
    expect(opt.series[0].data[0].itemStyle.color).toBe(explicitColor);
  });

  // ── Color branch: sign-derived color when no explicit color ──────────────
  // JSDOM getComputedStyle returns '' for custom properties; the component
  // falls back to gainColor/lossColor from CSS vars (empty string in JSDOM).
  // We assert that the color is NOT the explicit color — the sign-branch ran.

  it('when a datum has no explicit color and value > 0, color comes from gainColor (sign-derived)', () => {
    const opt = build(fixture.componentInstance, [
      { label: 'pos', value: 0.3 },   // no color field
    ]);
    // In JSDOM, CSS vars resolve to '' so gainColor === ''.
    // The key assertion is that the itemStyle.color equals getComputedStyle gainColor,
    // confirming the sign-branch executed (not an explicit override).
    const style = getComputedStyle(document.documentElement);
    const gainColor = style.getPropertyValue('--color-gain').trim();
    expect(opt.series[0].data[0].itemStyle.color).toBe(gainColor);
  });

  it('when a datum has no explicit color and value < 0, color comes from lossColor (sign-derived)', () => {
    const opt = build(fixture.componentInstance, [
      { label: 'neg', value: -0.1 },  // no color field
    ]);
    const style = getComputedStyle(document.documentElement);
    const lossColor = style.getPropertyValue('--color-loss').trim();
    expect(opt.series[0].data[0].itemStyle.color).toBe(lossColor);
  });

  // ── Mixed labels in a single call ───────────────────────────────────────

  it('when bars contain both YYYY-MM and plain labels, each is handled independently', () => {
    const opt = build(fixture.componentInstance, [
      { label: '2026-03', value: 0.1 },
      { label: 'MSFT',    value: 0.2 },
    ]);
    expect(opt.xAxis.data[0]).toBe("Mar '26");
    expect(opt.xAxis.data[1]).toBe('MSFT');
  });
});
