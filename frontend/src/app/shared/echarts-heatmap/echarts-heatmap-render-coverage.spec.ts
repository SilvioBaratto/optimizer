// Render-coverage spec for EchartsHeatmapComponent.
// The base spec (echarts-heatmap.spec.ts) already covers:
//   - N×N grid axes/cells, empty assets, ngOnDestroy, CHART_EXPORTABLE.
// This file adds the responsive branches not yet exercised:
//   - narrow (clientWidth=0 → isNarrow=true) → visualMap orient='horizontal'
//   - wide (clientWidth=700 → isNarrow=false) → visualMap orient='vertical'
//   - assets.length <= 10 → series[0].label.show = true
//   - assets.length > 10 → series[0].label.show = false
//   - dataZoom present when isNarrow AND assets.length > 10

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
} from '../../../testing';
import { EchartsHeatmapComponent } from './echarts-heatmap';

// ---------------------------------------------------------------------------
// Cast helper — mirrors the idiom in echarts-heatmap.spec.ts
// ---------------------------------------------------------------------------

interface HeatmapOption {
  series: Array<{ label: { show: boolean } }>;
  visualMap: { orient: string };
  dataZoom?: unknown[];
}

function build(
  comp: EchartsHeatmapComponent,
  assets: string[],
  matrix: number[][],
): HeatmapOption {
  return (
    comp as unknown as {
      buildOption(a: string[], m: number[][]): HeatmapOption;
    }
  ).buildOption(assets, matrix);
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** 3 assets — within the ≤10 label-show threshold */
const SMALL_ASSETS = ['AAPL', 'MSFT', 'GOOG'];
const SMALL_MATRIX = SMALL_ASSETS.map((_, i) =>
  SMALL_ASSETS.map((__, j) => (i === j ? 1 : 0.5)),
);

/** 12 assets — exceeds the >10 label-hide threshold */
const BIG_ASSETS = Array.from({ length: 12 }, (_, i) => `A${i}`);
const BIG_MATRIX = BIG_ASSETS.map((_, i) =>
  BIG_ASSETS.map((__, j) => (i === j ? 1 : 0.3)),
);

function stubWidth(
  comp: EchartsHeatmapComponent,
  width: number,
): void {
  // Access the private viewChild signal and patch nativeElement.clientWidth.
  const containerSignal = (
    comp as unknown as { container: () => { nativeElement: HTMLElement } }
  ).container();
  Object.defineProperty(containerSignal.nativeElement, 'clientWidth', {
    value: width,
    configurable: true,
  });
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsHeatmapComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsHeatmapComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsHeatmapComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsHeatmapComponent);
    // Required: resolve viewChild('container') before any buildOption cast.
    fixture.detectChanges();
  });

  // ── Narrow container → horizontal visualMap ──────────────────────────────
  // JSDOM initialises clientWidth to 0, which is < 500 → isNarrow = true.

  it('when container is narrow (clientWidth=0 < 500), visualMap orient is "horizontal"', () => {
    stubWidth(fixture.componentInstance, 0);
    const opt = build(fixture.componentInstance, SMALL_ASSETS, SMALL_MATRIX);
    expect(opt.visualMap.orient).toBe('horizontal');
  });

  // ── Wide container → vertical visualMap ─────────────────────────────────

  it('when container is wide (clientWidth=700 >= 500), visualMap orient is "vertical"', () => {
    stubWidth(fixture.componentInstance, 700);
    const opt = build(fixture.componentInstance, SMALL_ASSETS, SMALL_MATRIX);
    expect(opt.visualMap.orient).toBe('vertical');
  });

  // ── assets.length <= 10 → label shown ───────────────────────────────────

  it('when assets.length <= 10, series label.show is true', () => {
    stubWidth(fixture.componentInstance, 700);
    const opt = build(fixture.componentInstance, SMALL_ASSETS, SMALL_MATRIX);
    expect(opt.series[0].label.show).toBe(true);
  });

  // ── assets.length > 10 → label hidden ───────────────────────────────────

  it('when assets.length > 10, series label.show is false', () => {
    stubWidth(fixture.componentInstance, 700);
    const opt = build(fixture.componentInstance, BIG_ASSETS, BIG_MATRIX);
    expect(opt.series[0].label.show).toBe(false);
  });

  // ── dataZoom absent when wide ────────────────────────────────────────────
  // dataZoom requires isNarrow=true AND assets.length > 10.

  it('when container is wide, dataZoom is absent even for large asset sets', () => {
    stubWidth(fixture.componentInstance, 700);
    const opt = build(fixture.componentInstance, BIG_ASSETS, BIG_MATRIX);
    expect(opt.dataZoom).toBeUndefined();
  });

  // ── dataZoom present when narrow AND assets > 10 ────────────────────────

  it('when narrow AND assets.length > 10, dataZoom sliders are present', () => {
    stubWidth(fixture.componentInstance, 0);
    const opt = build(fixture.componentInstance, BIG_ASSETS, BIG_MATRIX);
    expect(Array.isArray(opt.dataZoom)).toBe(true);
    expect((opt.dataZoom as unknown[]).length).toBeGreaterThan(0);
  });

  // ── dataZoom absent when narrow BUT assets <= 10 ────────────────────────
  // useDataZoom = isNarrow && assets.length > 10; only large + narrow triggers it.

  it('when narrow BUT assets.length <= 10, dataZoom is absent', () => {
    stubWidth(fixture.componentInstance, 0);
    const opt = build(fixture.componentInstance, SMALL_ASSETS, SMALL_MATRIX);
    expect(opt.dataZoom).toBeUndefined();
  });
});
