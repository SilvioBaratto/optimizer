import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeStressScenarios } from '../../../testing';
import { StressPanelComponent } from './stress-panel';

describe('StressPanelComponent', () => {
  let fixture: ComponentFixture<StressPanelComponent>;
  let comp: StressPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [StressPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(StressPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when no scenario is selected, the waterfall is empty', () => {
    fixture.componentRef.setInput('scenarios', makeStressScenarios({ id: 's1' }));
    expect(comp.selectedScenario()).toBeNull();
    expect(comp.waterfallCategories()).toEqual([]);
  });

  it('when a scenario is selected, its sector waterfall is exposed', () => {
    fixture.componentRef.setInput('scenarios', makeStressScenarios({ id: 's1' }));
    comp.selectScenario('s1');
    expect(comp.selectedScenario()?.id).toBe('s1');
    expect(comp.waterfallCategories().length).toBe(8);
    expect(comp.waterfallValues().length).toBe(8);
  });

  it('when the selected scenario is clicked again, the selection toggles off', () => {
    fixture.componentRef.setInput('scenarios', makeStressScenarios({ id: 's1' }));
    comp.selectScenario('s1');
    comp.selectScenario('s1');
    expect(comp.selectedScenario()).toBeNull();
  });

  it('when scenarios are present, the comparison table maps one row each', () => {
    fixture.componentRef.setInput('scenarios', makeStressScenarios({ id: 's1' }));
    expect(comp.comparisonRows().length).toBe(1);
    expect(comp.comparisonRows()[0]['worstAsset']).toBe('TLT');
  });
});
