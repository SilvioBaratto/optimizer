// Render-coverage spec for EchartsHistogramComponent.
// The base spec (echarts-histogram.spec.ts) already covers:
//   - empty values → {} (no series), single bar series, overlayNormal 2-series,
//     zero-range guard (range===0 → single bar), ngOnDestroy, CHART_EXPORTABLE.
// This file adds the explicit-bins vs Sturges-default branch:
//   - When bins > 0, that exact count is used (explicit path).
//   - When bins === 0 (default), Sturges' formula ceil(log2(n)+1) is used.
//   - The two paths produce different bucket counts for the same dataset,
//     confirming both branches execute.

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
} from '../../../testing';
import { EchartsHistogramComponent } from './echarts-histogram';

// ---------------------------------------------------------------------------
// Cast helper — mirrors the idiom in echarts-histogram.spec.ts
// ---------------------------------------------------------------------------

interface HistogramOption {
  xAxis?: { data: string[] };
  series?: Array<{ data: unknown[] }>;
}

function build(comp: EchartsHistogramComponent, values: number[]): HistogramOption {
  return (comp as unknown as { buildOption(v: number[]): HistogramOption }).buildOption(values);
}

// 20 distinct values so Sturges gives ceil(log2(20)+1) = ceil(5.32) = 6 bins.
const VALUES_20 = Array.from({ length: 20 }, (_, i) => i * 0.01);

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsHistogramComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsHistogramComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsHistogramComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsHistogramComponent);
    fixture.detectChanges();
  });

  // ── Sturges default (bins input === 0) ───────────────────────────────────

  it('when bins input is 0 (default), Sturges formula determines bucket count', () => {
    // bins defaults to 0 — no setInput needed
    const opt = build(fixture.componentInstance, VALUES_20);

    const sturgesBins = Math.ceil(Math.log2(VALUES_20.length) + 1); // 6
    expect(opt.xAxis!.data.length).toBe(sturgesBins);
  });

  // ── Explicit bins path ───────────────────────────────────────────────────

  it('when bins input is set explicitly, that exact count is used', () => {
    fixture.componentRef.setInput('bins', 4);
    fixture.detectChanges();

    const opt = build(fixture.componentInstance, VALUES_20);
    expect(opt.xAxis!.data.length).toBe(4);
  });

  // ── Explicit bins differs from Sturges ───────────────────────────────────
  // Confirms the two branches produce different results for the same dataset.

  it('explicit bins=3 produces fewer buckets than Sturges default for the same data', () => {
    const sturgesOpt = build(fixture.componentInstance, VALUES_20);
    const sturgesBuckets = sturgesOpt.xAxis!.data.length;

    fixture.componentRef.setInput('bins', 3);
    fixture.detectChanges();

    const explicitOpt = build(fixture.componentInstance, VALUES_20);
    expect(explicitOpt.xAxis!.data.length).toBe(3);
    expect(explicitOpt.xAxis!.data.length).toBeLessThan(sturgesBuckets);
  });

  // ── Zero-range path (range === 0) ────────────────────────────────────────
  // Intentional minimal overlap with base spec to confirm the single-bucket
  // path produces exactly 1 xAxis label (the value itself).

  it('when all values are identical (range=0), xAxis has exactly one category', () => {
    const opt = build(fixture.componentInstance, [3.14, 3.14, 3.14]);
    // The zero-range path returns a single-element category axis
    expect(opt.xAxis!.data.length).toBe(1);
    expect(opt.xAxis!.data[0]).toBe((3.14).toFixed(4));
  });

  // ── overlayNormal: series count ──────────────────────────────────────────
  // Intentional minimal overlap: base spec uses VALUES (10 items), we use
  // VALUES_20 with std>0 to confirm the normal overlay appended the line series.

  it('when overlayNormal=true and std>0, a second line series is appended', () => {
    fixture.componentRef.setInput('overlayNormal', true);
    fixture.detectChanges();

    const opt = build(fixture.componentInstance, VALUES_20);
    expect(opt.series!.length).toBe(2);
  });
});
