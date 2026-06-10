// Render-coverage spec for EchartsRollingMetricsComponent.
// Focuses on the grid.right branch (single vs dual formatters) that the base
// spec does not assert, plus confirms the three @if template arms mount cleanly.
// The base spec (echarts-rolling-metrics.spec.ts) already covers y-axis count,
// axis formatters, height binding, and CHART_EXPORTABLE — those are not
// duplicated here.

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
} from '../../../testing';
import {
  EchartsRollingMetricsComponent,
  type RollingMetricSeries,
} from './echarts-rolling-metrics';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRatioSeries(name = 'A'): RollingMetricSeries {
  return { name, formatter: 'ratio', values: [{ date: '2026-01-01', value: 1.0 }] };
}

function makePercentSeries(name = 'B'): RollingMetricSeries {
  return { name, formatter: 'percent', values: [{ date: '2026-01-01', value: 0.1 }] };
}

function gridRight(comp: EchartsRollingMetricsComponent): number {
  const option = comp.buildOptionForTest();
  return (option['grid'] as { right: number }).right;
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('EchartsRollingMetricsComponent — render-coverage', () => {
  let fixture: ComponentFixture<EchartsRollingMetricsComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports: [EchartsRollingMetricsComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
  });

  // ── Template arm: loading ────────────────────────────────────────────────

  it('when loading=true, renders [data-testid="loading"] and no other state', () => {
    fixture.componentRef.setInput('series', [makeRatioSeries()]);
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();

    const root: HTMLElement = fixture.nativeElement;
    expect(root.querySelector('[data-testid="loading"]')).not.toBeNull();
    expect(root.querySelector('[data-testid="empty"]')).toBeNull();
    expect(root.querySelector('[data-testid="chart"]')).toBeNull();
  });

  // ── Template arm: empty (all values arrays empty) ───────────────────────

  it('when loading=false and all series have empty values, renders [data-testid="empty"]', () => {
    const emptySeries: RollingMetricSeries[] = [
      { name: 'x', formatter: 'ratio', values: [] },
    ];
    fixture.componentRef.setInput('series', emptySeries);
    fixture.componentRef.setInput('loading', false);
    fixture.detectChanges();

    const root: HTMLElement = fixture.nativeElement;
    expect(root.querySelector('[data-testid="empty"]')).not.toBeNull();
    expect(root.querySelector('[data-testid="chart"]')).toBeNull();
  });

  // ── Template arm: chart ──────────────────────────────────────────────────

  it('when loading=false and at least one series has values, renders [data-testid="chart"]', () => {
    fixture.componentRef.setInput('series', [makeRatioSeries()]);
    fixture.componentRef.setInput('loading', false);
    fixture.detectChanges();

    const root: HTMLElement = fixture.nativeElement;
    expect(root.querySelector('[data-testid="chart"]')).not.toBeNull();
    expect(root.querySelector('[data-testid="empty"]')).toBeNull();
    expect(root.querySelector('[data-testid="loading"]')).toBeNull();
  });

  // ── grid.right branch: single distinct formatter → 16 ───────────────────

  it('when all series share the same formatter, grid.right equals 16 (single y-axis layout)', () => {
    fixture.componentRef.setInput('series', [makeRatioSeries('S1'), makeRatioSeries('S2')]);
    fixture.detectChanges();

    expect(gridRight(fixture.componentInstance)).toBe(16);
  });

  // ── grid.right branch: dual distinct formatters → 50 ────────────────────

  it('when series have two distinct formatters, grid.right equals 50 (dual y-axis layout)', () => {
    fixture.componentRef.setInput('series', [makeRatioSeries(), makePercentSeries()]);
    fixture.detectChanges();

    expect(gridRight(fixture.componentInstance)).toBe(50);
  });
});
