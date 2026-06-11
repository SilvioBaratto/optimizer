/**
 * taa-panel-render-coverage.spec.ts — issue #943
 *
 * Render-coverage for TaaPanelComponent. Drives the macro-calibration detail
 * @if, the signal-table rows (title-cased, all four regimes) and mounts the bar
 * + line ECharts (afterNextRender) so buildBarOption / buildLineOption run.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
  makeTAASignal,
  makeFactorReturnSeries,
  makeMacroCalibrationApiResponse,
} from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { TaaPanelComponent } from './taa-panel';

async function mount(
  inputs: Record<string, unknown> = {},
): Promise<ComponentFixture<TaaPanelComponent>> {
  installResizeObserverStub();
  await configureTestBed({ imports: [TaaPanelComponent], withHttp: false, providers: [ICON_PROVIDER] });
  const fixture = TestBed.createComponent(TaaPanelComponent);
  for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
  fixture.detectChanges();
  await fixture.whenStable(); // afterNextRender → init charts → build*Option runs
  fixture.detectChanges();
  return fixture;
}

describe('TaaPanelComponent — render-coverage (#943)', () => {
  it('when populated, the signal table and both chart containers render', async () => {
    const fixture = await mount({
      signals: [makeTAASignal()],
      factorReturns: [makeFactorReturnSeries()],
    });
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('app-data-table')).not.toBeNull();
    expect(el.textContent).toContain('Current vs Tilted Weights');
    expect(el.textContent).toContain('Factor Return History');
  });

  it('when macroCalibration is null, the detail card is absent', async () => {
    const fixture = await mount({});
    expect((fixture.nativeElement as HTMLElement).textContent).not.toContain('risk aversion');
  });

  it('when macroCalibration is set, the detail card renders its fields', async () => {
    const fixture = await mount({ macroCalibration: makeMacroCalibrationApiResponse() });
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('risk aversion');
    expect(text).toContain('Confidence');
  });

  it('signalRows title-cases the factor name and maps all four regimes', async () => {
    const signals = (['expansion', 'slowdown', 'recession', 'recovery'] as const).map((regime) =>
      makeTAASignal({ regime, factor: 'momentum' }),
    );
    const fixture = await mount({ signals });
    const rows = fixture.componentInstance.signalRows();
    expect(rows.length).toBe(4);
    expect(rows[0]['factor']).toBe('Momentum');
    expect(rows.map((r) => r['regime'])).toEqual(['expansion', 'slowdown', 'recession', 'recovery']);
  });

  it('when there are no signals, signalRows is empty', async () => {
    const fixture = await mount({});
    expect(fixture.componentInstance.signalRows()).toEqual([]);
  });

  it('updating data on wide containers re-runs the chart effects with the wide-layout branch', async () => {
    const fixture = await mount({ signals: [makeTAASignal()], factorReturns: [makeFactorReturnSeries()] });
    const el = fixture.nativeElement as HTMLElement;

    // Chart init (afterNextRender → dynamic import → echarts.init) is async; poll
    // until ECharts has tagged the containers before driving the effects so the
    // `barChart && …` / `lineChart && …` true branches are taken.
    for (let i = 0; i < 100 && el.querySelectorAll('[_echarts_instance_]').length < 2; i++) {
      await new Promise((r) => setTimeout(r, 10));
      fixture.detectChanges();
    }
    // clientWidth undefined exercises the `?? 600` width default (and the
    // resulting non-narrow grid arm) when the effects re-run build*Option.
    el.querySelectorAll('div.w-full').forEach((d) =>
      Object.defineProperty(d, 'clientWidth', { value: undefined, configurable: true }),
    );
    fixture.componentRef.setInput('signals', [makeTAASignal({ factor: 'value' }), makeTAASignal()]);
    fixture.componentRef.setInput('factorReturns', [
      makeFactorReturnSeries(),
      makeFactorReturnSeries({ factor: 'book_to_price' }),
    ]);
    fixture.detectChanges();

    expect(fixture.componentInstance.signalRows().length).toBe(2);
  });
});
