import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { FactorPanelComponent } from './factor-panel';
import type { FactorExposure } from '../../models/risk.model';

function exposure(factor: string, contribution: number): FactorExposure {
  return { factor, exposure: contribution, contribution, marginalContribution: contribution };
}

describe('FactorPanelComponent', () => {
  let fixture: ComponentFixture<FactorPanelComponent>;
  let comp: FactorPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [FactorPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(FactorPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when exposures are empty, specificRisk is the full 1 and the donut is empty', () => {
    expect(comp.specificRisk()).toBe(1);
    expect(comp.donutData()).toEqual([]);
  });

  it('when contributions sum below 1, specificRisk is the linear remainder', () => {
    fixture.componentRef.setInput('exposures', [exposure('value', 0.3), exposure('momentum', 0.2)]);
    expect(comp.specificRisk()).toBeCloseTo(0.5); // 1 - 0.5
  });

  it('when contributions exceed 1, specificRisk is clamped at 0', () => {
    fixture.componentRef.setInput('exposures', [exposure('value', 0.7), exposure('momentum', 0.5)]);
    expect(comp.specificRisk()).toBe(0); // max(0, 1 - 1.2)
  });

  it('when some contributions are negative, the donut keeps only positives', () => {
    fixture.componentRef.setInput('exposures', [exposure('value', 0.4), exposure('momentum', -0.2)]);
    expect(comp.donutData().length).toBe(1);
    expect(comp.donutData()[0].label).toBe('value');
  });
});
