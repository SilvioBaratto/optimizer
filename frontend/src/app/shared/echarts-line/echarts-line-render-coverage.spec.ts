// Render-coverage spec for EchartsLineComponent.
// The base spec (echarts-line.spec.ts) covers mount, empty/populated data,
// ngOnDestroy, and CHART_EXPORTABLE token.
// This file adds branches not yet exercised:
//   - areaFill true → series areaStyle is defined; false → areaStyle undefined.
//   - yAxisPrefix non-empty → formatter uses prefix, no trailing '%'.
//   - yAxisPrefix empty (default) → formatter appends '%'.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { EChartsOption } from 'echarts';

import {
  configureTestBed,
  installResizeObserverStub,
  getSeriesAt,
} from '../../../testing';
import { EchartsLineComponent } from './echarts-line';

// ---------------------------------------------------------------------------
// Cast helper — buildOption is private; cast breaks the encapsulation only in
// tests so the real component remains unmodified.
// ---------------------------------------------------------------------------

function build(
  comp: EchartsLineComponent,
  labels: string[],
  values: number[],
): EChartsOption {
  return (
    comp as unknown as {
      buildOption(l: string[], v: number[]): EChartsOption;
    }
  ).buildOption(labels, values);
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsLineComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsLineComponent>;
  let comp: EchartsLineComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [EchartsLineComponent], withHttp: false });
    fixture = TestBed.createComponent(EchartsLineComponent);
    // detectChanges first so viewChild('container') resolves before the cast
    fixture.detectChanges();
    comp = fixture.componentInstance;
  });

  // ── areaFill branch ───────────────────────────────────────────────────────

  it('when areaFill is true, series[0].areaStyle is defined', () => {
    fixture.componentRef.setInput('areaFill', true);
    const opt = build(comp, ['Jan', 'Feb'], [1, 2]);
    const series = getSeriesAt(opt, 0);
    expect(series['areaStyle']).toBeDefined();
    expect(series['areaStyle']).not.toBeUndefined();
  });

  it('when areaFill is false (default), series[0].areaStyle is undefined', () => {
    fixture.componentRef.setInput('areaFill', false);
    const opt = build(comp, ['Jan', 'Feb'], [1, 2]);
    const series = getSeriesAt(opt, 0);
    expect(series['areaStyle']).toBeUndefined();
  });

  // ── yAxisPrefix branch — formatter ────────────────────────────────────────

  it('when yAxisPrefix is "$", formatter prefixes the value and omits "%"', () => {
    fixture.componentRef.setInput('yAxisPrefix', '$');
    const opt = build(comp, ['Jan', 'Feb'], [1, 2]);
    const yAxis = opt.yAxis as { axisLabel: { formatter: (v: number) => string } };
    const result = yAxis.axisLabel.formatter(1.5);
    expect(result).toBe('$1.50');
    expect(result).not.toContain('%');
  });

  it('when yAxisPrefix is empty (default), formatter appends "%"', () => {
    fixture.componentRef.setInput('yAxisPrefix', '');
    const opt = build(comp, ['Jan', 'Feb'], [1, 2]);
    const yAxis = opt.yAxis as { axisLabel: { formatter: (v: number) => string } };
    const result = yAxis.axisLabel.formatter(1.5);
    expect(result).toBe('1.50%');
  });

  it('tooltip formatter uses yAxisPrefix when set', () => {
    fixture.componentRef.setInput('yAxisPrefix', '€');
    const opt = build(comp, ['Jan'], [3]);
    const tooltip = opt.tooltip as {
      formatter: (params: Array<{ name: string; value: number }>) => string;
    };
    const html = tooltip.formatter([{ name: 'Jan', value: 3.0 }]);
    expect(html).toContain('€3.00');
    expect(html).not.toContain('%');
  });

  it('tooltip formatter appends "%" when yAxisPrefix is empty', () => {
    fixture.componentRef.setInput('yAxisPrefix', '');
    const opt = build(comp, ['Jan'], [3]);
    const tooltip = opt.tooltip as {
      formatter: (params: Array<{ name: string; value: number }>) => string;
    };
    const html = tooltip.formatter([{ name: 'Jan', value: 3.0 }]);
    expect(html).toContain('3.00%');
  });

  // ── Series type always 'line' ─────────────────────────────────────────────

  it('series[0].type is always "line"', () => {
    const opt = build(comp, ['Jan', 'Feb', 'Mar'], [1, 2, 3]);
    expect(getSeriesAt(opt, 0)['type']).toBe('line');
  });
});
