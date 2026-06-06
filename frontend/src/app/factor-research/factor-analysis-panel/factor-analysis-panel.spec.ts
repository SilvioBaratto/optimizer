import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub } from '../../../testing';
import { FactorAnalysisPanelComponent } from './factor-analysis-panel';
import type { FactorReturnSeries, FactorValidateResponse } from '../factor.model';

function series(factor: 'momentum_12_1' | 'volatility', up: boolean): FactorReturnSeries {
  return {
    factor,
    group: 'momentum',
    points: [
      { date: '2026-01-01', cumReturn: 0 },
      { date: '2026-02-01', cumReturn: up ? 1 : -1 },
      { date: '2026-03-01', cumReturn: up ? 2 : -2 },
    ],
  };
}

function validateReport(pValue: number): FactorValidateResponse {
  return {
    report_date: '2026-01-01',
    factor_type: 'momentum_12_1',
    validation_type: 'in_sample',
    ic_mean: 0.05,
    ic_std: 0.1,
    icir: 0.5,
    t_stat: 2,
    p_value: pValue,
    vif: 1.1,
    details: null,
  };
}

describe('FactorAnalysisPanelComponent', () => {
  let fixture: ComponentFixture<FactorAnalysisPanelComponent>;
  let comp: FactorAnalysisPanelComponent;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [FactorAnalysisPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(FactorAnalysisPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when two identical series are given, their correlation is 1; the diagonal is 1', () => {
    fixture.componentRef.setInput('factorReturns', [series('momentum_12_1', true), series('volatility', true)]);
    const m = comp.correlationMatrix();
    expect(m[0][0]).toBe(1);
    expect(m[0][1]).toBe(1);
  });

  it('when two anti-correlated series are given, their correlation is -1', () => {
    fixture.componentRef.setInput('factorReturns', [series('momentum_12_1', true), series('volatility', false)]);
    expect(comp.correlationMatrix()[0][1]).toBe(-1);
  });

  it('when validate p-value is below 0.05, the row is marked significant', () => {
    fixture.componentRef.setInput('validateReport', validateReport(0.01));
    expect(comp.validateRows()[0]['significant']).toBe('true');
  });

  it('when validate p-value is at or above 0.05, the row is not significant', () => {
    fixture.componentRef.setInput('validateReport', validateReport(0.2));
    expect(comp.validateRows()[0]['significant']).toBe('false');
  });

  it('when there is no validate report, the rows are empty', () => {
    expect(comp.validateRows()).toEqual([]);
  });

  it('when submitted, validate and runQuintileSpread emit the form factor', () => {
    fixture.componentRef.setInput('factorName', 'momentum_12_1');
    let validated: string | undefined;
    let quintile: string | undefined;
    comp.validate.subscribe((f) => (validated = f));
    comp.runQuintileSpread.subscribe((f) => (quintile = f));
    comp.submitValidate();
    comp.submitQuintile();
    expect(validated).toBe('momentum_12_1');
    expect(quintile).toBe('momentum_12_1');
  });
});
