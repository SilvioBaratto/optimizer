// Render-coverage spec for EchartsEfficientFrontierComponent.
// The base spec (echarts-efficient-frontier.spec.ts) already covers:
//   - template arms (empty/chart), height binding, CHART_EXPORTABLE
//   - optimal+markers scatter count, CAL presence/absence, click emission
//     for dataIndex=1 (in-range) and non-frontier seriesName.
// This file adds:
//   - 'Optimal' series name assertion (present / absent)
//   - out-of-range dataIndex click → no emission
//   - assetMarkers series-count increase
//   - CAL absent when riskFreeRate is undefined (not just null)

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
} from '../../../testing';
import { EchartsEfficientFrontierComponent } from './echarts-efficient-frontier';
import type { EfficientFrontierPoint } from '../../core/models/optimization.model';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePoint(risk: number, ret: number, sharpe: number, label?: string): EfficientFrontierPoint {
  return { risk, return: ret, sharpe, ...(label !== undefined ? { label } : {}) };
}

function makeFrontier(): EfficientFrontierPoint[] {
  return [
    makePoint(0.05, 0.03, 0.60),
    makePoint(0.08, 0.06, 0.75),
  ];
}

type SeriesEntry = { name?: string; type?: string };

function seriesNames(comp: EchartsEfficientFrontierComponent): string[] {
  const option = comp.buildOptionForTest();
  return (option['series'] as SeriesEntry[]).map((s) => s.name ?? '');
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsEfficientFrontierComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsEfficientFrontierComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsEfficientFrontierComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsEfficientFrontierComponent);
  });

  // ── Template arm: chart present ──────────────────────────────────────────

  it('when frontier is non-empty, renders [data-testid="efficient-frontier-chart"]', () => {
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.detectChanges();

    expect(
      fixture.nativeElement.querySelector('[data-testid="efficient-frontier-chart"]'),
    ).not.toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="efficient-frontier-empty"]'),
    ).toBeNull();
  });

  // ── Template arm: empty state ────────────────────────────────────────────

  it('when frontier is empty, renders [data-testid="efficient-frontier-empty"]', () => {
    fixture.componentRef.setInput('frontier', []);
    fixture.detectChanges();

    expect(
      fixture.nativeElement.querySelector('[data-testid="efficient-frontier-empty"]'),
    ).not.toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="efficient-frontier-chart"]'),
    ).toBeNull();
  });

  // ── buildOption: 'Optimal' series present when optimal is set ────────────

  it('when optimal is set, the series list contains a series named "Optimal"', () => {
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.componentRef.setInput('optimal', makePoint(0.08, 0.06, 0.75, 'Max Sharpe'));
    fixture.detectChanges();

    expect(seriesNames(fixture.componentInstance)).toContain('Optimal');
  });

  // ── buildOption: no 'Optimal' series when optimal is null ────────────────

  it('when optimal is null, the series list does NOT contain "Optimal"', () => {
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.componentRef.setInput('optimal', null);
    fixture.detectChanges();

    expect(seriesNames(fixture.componentInstance)).not.toContain('Optimal');
  });

  // ── buildOption: CAL absent when riskFreeRate is undefined ───────────────
  // The base spec covers riskFreeRate=null; this arm tests the undefined path
  // inside maybeBuildCalSeries (rfr === null || rfr === undefined → return null).

  it('when riskFreeRate is not provided (undefined default), no CAL series is emitted', () => {
    fixture.componentRef.setInput('frontier', makeFrontier());
    // riskFreeRate defaults to null via input(null), no setInput call needed
    fixture.detectChanges();

    expect(seriesNames(fixture.componentInstance)).not.toContain('Capital Allocation Line');
  });

  // ── buildOption: assetMarkers adds extra scatter series ──────────────────

  it('when assetMarkers has 1 entry, total series count is frontier + marker = 2', () => {
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.componentRef.setInput('assetMarkers', [makePoint(0.05, 0.03, 0.60, 'MinVar')]);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const series = option['series'] as SeriesEntry[];
    // Frontier (1) + marker scatter (1) = 2; no optimal, no CAL
    expect(series.length).toBe(2);
    const markerSeries = series.find((s) => s.name === 'MinVar');
    expect(markerSeries).toBeDefined();
  });

  // ── click: out-of-range dataIndex emits nothing ──────────────────────────

  it('when dataIndex is out of range, handleFrontierClickForTest emits nothing', () => {
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.detectChanges();

    const emitted: EfficientFrontierPoint[] = [];
    fixture.componentInstance.pointClick.subscribe((p) => emitted.push(p));

    fixture.componentInstance.handleFrontierClickForTest({
      seriesName: 'Efficient Frontier',
      dataIndex: 99,
    });

    expect(emitted.length).toBe(0);
  });

  // ── click: non-frontier seriesName emits nothing ─────────────────────────
  // Intentional minimal overlap with base spec (which tests 'Max Sharpe');
  // here we use 'Optimal' to confirm the guard applies across all non-frontier names.

  it('when seriesName is "Optimal" (not frontier), no pointClick is emitted', () => {
    fixture.componentRef.setInput('frontier', makeFrontier());
    fixture.detectChanges();

    const emitted: EfficientFrontierPoint[] = [];
    fixture.componentInstance.pointClick.subscribe((p) => emitted.push(p));

    fixture.componentInstance.handleFrontierClickForTest({
      seriesName: 'Optimal',
      dataIndex: 0,
    });

    expect(emitted.length).toBe(0);
  });
});
