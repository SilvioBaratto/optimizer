import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { PreprocessingPanelComponent } from './preprocessing-panel';

describe('PreprocessingPanelComponent', () => {
  let fixture: ComponentFixture<PreprocessingPanelComponent>;
  let comp: PreprocessingPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [PreprocessingPanelComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(PreprocessingPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when the returns type is set, the signal updates', () => {
    comp.setReturnsType('log');
    expect(comp.returnsType()).toBe('log');
  });

  it('when the frequency is set, the signal updates', () => {
    comp.setFrequency('weekly');
    expect(comp.frequency()).toBe('weekly');
  });

  it('when the missing-data policy is set, the signal updates', () => {
    comp.setMissingData('interpolate');
    expect(comp.missingData()).toBe('interpolate');
  });

  it('when the outlier treatment is set, the signal updates', () => {
    comp.setOutlierTreatment('trim');
    expect(comp.outlierTreatment()).toBe('trim');
  });

  it('when the lookback years are set, the signal updates', () => {
    comp.setLookbackYears(10);
    expect(comp.lookbackYears()).toBe(10);
  });
});
