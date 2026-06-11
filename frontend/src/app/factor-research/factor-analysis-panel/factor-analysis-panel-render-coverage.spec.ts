/**
 * factor-analysis-panel-render-coverage.spec.ts — issue #943
 *
 * Render-coverage for FactorAnalysisPanelComponent. Drives the validate-rows
 * and quintile-spread @if/@for arms, the IC report table, the correlation
 * computeds (pairwise correlation over factor returns) and the validate /
 * quintile output emissions. The line ECharts is mounted so buildLineOption runs.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  configureTestBed,
  installResizeObserverStub,
  makeFactorICReport,
  makeFactorReturnSeries,
  makeFactorValidateResponse,
} from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { FactorAnalysisPanelComponent } from './factor-analysis-panel';
import type { FactorQuintileSpreadApiResponse } from '../factor.model';

function quintile(): FactorQuintileSpreadApiResponse {
  return {
    quintile_cumulative_returns: { Q1: [0.0, 0.01], Q5: [0.0, 0.03] },
    spread_cumulative_return: [0.0, 0.02, 0.04],
    annualized_spread: 0.12,
  };
}

async function mount(
  inputs: Record<string, unknown> = {},
): Promise<ComponentFixture<FactorAnalysisPanelComponent>> {
  installResizeObserverStub();
  await configureTestBed({
    imports: [FactorAnalysisPanelComponent],
    withHttp: false,
    providers: [ICON_PROVIDER],
  });
  const fixture = TestBed.createComponent(FactorAnalysisPanelComponent);
  for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
  fixture.detectChanges();
  await fixture.whenStable();
  fixture.detectChanges();
  return fixture;
}

function button(el: HTMLElement, text: string): HTMLButtonElement {
  return Array.from(el.querySelectorAll('button')).find((b) => b.textContent?.includes(text)) as HTMLButtonElement;
}

describe('FactorAnalysisPanelComponent — render-coverage (#943)', () => {
  it('when fully populated, the IC/ICIR table, quintile card and heatmap render', async () => {
    const fixture = await mount({
      factorReturns: [makeFactorReturnSeries({ factor: 'momentum_12_1' }), makeFactorReturnSeries({ factor: 'book_to_price', group: 'value' })],
      icReports: [makeFactorICReport()],
      validateReport: makeFactorValidateResponse(),
      quintileSpread: quintile(),
    });
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('IC / ICIR report');
    expect(el.textContent).toContain('Quintile spread');
    expect(el.textContent).toContain('Q1');
    expect(el.querySelector('app-echarts-heatmap')).not.toBeNull();
    expect(el.querySelectorAll('app-data-table').length).toBeGreaterThanOrEqual(2);
  });

  it('when validateReport is null, the IC/ICIR table is absent', async () => {
    const fixture = await mount({ validateReport: null });
    expect((fixture.nativeElement as HTMLElement).textContent).not.toContain('IC / ICIR report');
  });

  it('when quintileSpread is null, the spread card is absent and the spread computeds return empties', async () => {
    const fixture = await mount({ quintileSpread: null });
    const comp = fixture.componentInstance;
    expect((fixture.nativeElement as HTMLElement).textContent).not.toContain('Quintile spread');
    expect(comp.quintileLabels()).toEqual([]);
    expect(comp.spreadSeries()).toEqual([]);
    expect(comp.annualizedSpread()).toBeNull();
  });

  it('validateRows marks significance from the p-value', async () => {
    const fixture = await mount({ validateReport: makeFactorValidateResponse({ p_value: 0.01 }) });
    expect(fixture.componentInstance.validateRows()[0]['significant']).toBe('true');
  });

  it('validateRows falls back to defaults and marks not-significant when fields are null', async () => {
    const fixture = await mount({
      validateReport: makeFactorValidateResponse({
        factor_type: null, ic_mean: null, ic_std: null, icir: null,
        t_stat: null, p_value: null, vif: null,
      }),
    });
    const row = fixture.componentInstance.validateRows()[0];
    expect(row['factor']).toBe('—');
    expect(row['icMean']).toBe(0);
    expect(row['significant']).toBe('false');
  });

  it('updating factorReturns after init re-renders the line chart', async () => {
    const fixture = await mount({ factorReturns: [makeFactorReturnSeries()] });
    fixture.componentRef.setInput('factorReturns', [
      makeFactorReturnSeries(),
      makeFactorReturnSeries({ factor: 'book_to_price' }),
    ]);
    fixture.detectChanges();
    await fixture.whenStable();
    expect(fixture.componentInstance.correlationMatrix().length).toBe(2);
  });

  it('correlationMatrix is 0 off-diagonal for a constant (zero-variance) series', async () => {
    const fixture = await mount({
      factorReturns: [
        makeFactorReturnSeries({ points: [{ date: 'a', cumReturn: 0.5 }, { date: 'b', cumReturn: 0.5 }] }),
        makeFactorReturnSeries({ factor: 'book_to_price' }),
      ],
    });
    expect(fixture.componentInstance.correlationMatrix()[0][1]).toBe(0);
  });

  it('icRows is empty when there are no IC reports', async () => {
    const fixture = await mount({ icReports: [] });
    expect(fixture.componentInstance.icRows()).toEqual([]);
  });

  it('correlationMatrix produces a unit-diagonal matrix sized to the factor count', async () => {
    const fixture = await mount({
      factorReturns: [makeFactorReturnSeries(), makeFactorReturnSeries({ factor: 'book_to_price' })],
    });
    const m = fixture.componentInstance.correlationMatrix();
    expect(m.length).toBe(2);
    expect(m[0][0]).toBe(1);
    expect(fixture.componentInstance.correlationAssets().length).toBe(2);
  });

  it('correlationMatrix yields 0 off-diagonal when a series has fewer than 2 points', async () => {
    const fixture = await mount({
      factorReturns: [
        makeFactorReturnSeries({ points: [{ date: '2026-01-01', cumReturn: 0.0 }] }),
        makeFactorReturnSeries({ factor: 'book_to_price' }),
      ],
    });
    expect(fixture.componentInstance.correlationMatrix()[0][1]).toBe(0);
  });

  it('clicking Run validate emits the form factor', async () => {
    const fixture = await mount({});
    let emitted: string | undefined;
    fixture.componentInstance.validate.subscribe((f) => (emitted = f));
    button(fixture.nativeElement, 'Run validate').click();
    expect(emitted).toBe('momentum_12_1');
  });

  it('clicking Run quintile spread emits the form factor', async () => {
    const fixture = await mount({});
    let emitted: string | undefined;
    fixture.componentInstance.runQuintileSpread.subscribe((f) => (emitted = f));
    button(fixture.nativeElement, 'Run quintile spread').click();
    expect(emitted).toBe('momentum_12_1');
  });
});
